"""
Neural network models to be used, mainly focused on ANN, CNN and LSTM
"""

import torch
import torch.nn as nn

#* ANN
class ANN(nn.Module):
    """
    A customizable Multilayer Perceptron for regression with Multiple fully-connected layers, batch normalization, dropout regularization and ReLU activation

    Args:
        - input_size (int): Number of input features
        - drop (float): Dropout probability

    Returns:
        - x (torch.Tensor): Output predition of shape [batch_size, 1]

    """

    def __init__(self, input_size, drop):
        super(ANN, self).__init__()

        self.fc_layers = nn.ModuleList([        # Fully Connected Layers with Sequential dimensionality reduction
            nn.Linear(input_size, 320),
            nn.Linear(320, 320),
            nn.Linear(320, 160),
            nn.Linear(160, 160),
            nn.Linear(160, 80),
            nn.Linear(80,40),
            nn.Linear(40, 20),
            nn.Linear(20, 10),
            nn.Linear(10, 1)
        ])

        self.drop_layers = nn.ModuleList([        # Regularization Layers
            nn.Dropout(drop),
            nn.Dropout(drop),
            nn.Dropout(drop),
            nn.Dropout(drop),
            nn.Dropout(drop),
            nn.Dropout(drop),
            nn.Dropout(drop),
        ])

        self.bn_layers = nn.ModuleList([        # Batch normalization for all hidden layers
            nn.BatchNorm1d(320),
            nn.BatchNorm1d(320),
            nn.BatchNorm1d(160),
            nn.BatchNorm1d(160),
            nn.BatchNorm1d(80),
            nn.BatchNorm1d(40),
            nn.BatchNorm1d(20),
        ])

        self.activation = nn.ReLU()        # Activation function

    def forward(self, x):
        """
        Forward pass through the network

        """
        for i in range(len(self.fc_layers) - 1):        # Process all hidden layers
            fc = self.fc_layers[i]
            x = fc(x)
            if i < len(self.bn_layers):        # Apply BN only if available
                x = self.bn_layers[i](x)
            x = self.activation(x)
            if i < len(self.drop_layers):        # Apply dropout only if available
                x = self.drop_layers[i](x)
        x = self.fc_layers[-1](x)        # Final layer
        return x
                

#* CNN
class ResBlock(nn.Module):
    def __init__(self, ni, no, kernel, pad):
        super(ResBlock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(ni, no, kernel, padding = pad),
            nn.LeakyReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(no, no, kernel, padding = pad),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        return x1 + x2
    

class CNN(nn.Module):
    def __init__(self, input_size, kernel_cnn, drop):
        super(CNN, self).__init__()
        pad = kernel_cnn // 2

        self.conv_blocks = nn.Sequential(
            ResBlock(input_size, 64, kernel_cnn, pad),  
            nn.MaxPool1d(2, 2),
            nn.Dropout(drop),
            
            ResBlock(64, 128, kernel_cnn, pad),        
            nn.MaxPool1d(2, 2),
            nn.Dropout(drop),
            
            ResBlock(128, 256, kernel_cnn, pad),      
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(drop)
        )

        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(256 + input_size, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
    

    def forward(self, x):

        x_last = x[:, :, -1]
        x = self.conv_blocks(x)
        x = x.mean(-1)
        x = torch.cat([x, x_last], dim = 1)
        x = self.fc(x)
        
        return x


#* LSTM
class LSTM(nn.Module):
    """
    Hybrid LSTM-GRU model for temporal prediction.

    Args:
        - input_size (int): Number of input features
        - drop (float): Dropout probability
        - n_neurons (int): Base neurons for recurrent layers (default = 192)
        - delay (int): Number of time steps to consider (time window)
        - num_layer_lstm (int): Number of recurrent layers
    """
    def __init__(self, input_size, drop, delay, num_layer_lstm, n_neurons = 192):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(        # LSTM layer
            input_size = input_size,
            hidden_size = n_neurons,
            num_layers = num_layer_lstm,
            bidirectional = True,
            batch_first = True
        )
        
        self.gru = nn.GRU(        # GRU layer
            input_size = n_neurons * 2,        # x2 bidirectional
            hidden_size = n_neurons * 3,
            bidirectional = True,
            batch_first = True
        )
        
        self.dense = nn.Sequential(        # Dense layers
            nn.Linear(n_neurons * 6 * delay, 96),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.drop = nn.Dropout(drop)        # Dropout

    def forward(self, x):
        lstm_out, _ = self.lstm(x)        # LSTM
        lstm_out = self.drop(lstm_out)
        
        gru_out, _ = self.gru(lstm_out)        # GRU
        gru_out = self.drop(gru_out)
        
        gru_flat = gru_out.reshape(gru_out.size(0), -1)        # Flatten + Dense (Flatten equivalent)
        return self.dense(gru_flat)
    


