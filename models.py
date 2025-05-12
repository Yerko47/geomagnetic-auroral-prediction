"""
Neural network models to be used, mainly focused on ANN, CNN and LSTM
"""

import torch
import torch.nn as nn

#* ANN
class ANN(nn.Module):
    """
    A customizable Multilayer Perceptron (MLP) for regression with multiple fully-connected layers, 
    batch normalization, dropout regularization, and ReLU activation.

    Args
        input_size : int
            Number of input features.
        drop : float
            Dropout probability for regularization.

    Returns
        x : torch.Tensor
            Output prediction of shape [batch_size, 1].
    """

    def __init__(self, input_size, drop):
        super(ANN, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(drop),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(drop),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(drop),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(drop),

            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(drop),

            nn.ReLU(),
            nn.Linear(16, 1)
        )


    def forward(self, x):
        return self.fc_layers(x)
                

#* CNN
class ResBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_cnn, stride, padd):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(input_size, output_size, kernel_cnn, stride, padd), 
                                    nn.LeakyReLU())
        
        self.layer2 = nn.Sequential(nn.Conv1d(output_size, output_size, kernel_cnn, stride, padd), 
                                    nn.LeakyReLU())

        self.downsample = nn.Sequential(
            nn.Conv1d(input_size, output_size, kernel_size=1, padding=0), 
            nn.BatchNorm1d(output_size), 
            nn.ReLU())

    def forward(self, x):
        identity = self.downsample(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        if x2.size() != identity.size():
            diff = x2.size(2) - identity.size(2)
            if diff > 0:
                identity = nn.functional.pad(identity, (diff // 2, diff - diff//2))
            elif diff < 0:
                x2 = nn.functional.pad(x2, (-diff//2, -diff + diff//2))

        return x2 + identity

class CNN(nn.Module):
    def __init__(self, input_size, kernel_cnn, drop, delay):
        super(CNN, self).__init__()
        padd = (kernel_cnn // 2) + 1 if (kernel_cnn // 2) % 2 == 0 else (kernel_cnn // 2)

        self.conv1 = ResBlock(input_size, 64, kernel_cnn, 2, padd)
        self.conv2 = ResBlock(64, 128, kernel_cnn, 2, padd)
        self.conv3 = ResBlock(128, 256, kernel_cnn, 2, padd)
        self.conv4 = ResBlock(256, 384, kernel_cnn, 2, padd)
        self.max_pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(drop)
        # Modificar la capa lineal para manejar correctamente las dimensiones
        self.fc1 = nn.Sequential(nn.Linear(384 + input_size, 64), nn.ReLU())
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [batch_size, input_size, delay]
        x_last = x[:, :, -1]  # [batch_size, input_size]
        
        x = self.conv1(x)     # [batch_size, 64, T]
        x = self.max_pool(x)
        x = self.dropout(x)
        
        x = self.conv2(x)     # [batch_size, 128, T]
        x = self.max_pool(x)
        x = self.dropout(x)
        
        x = self.conv3(x)     # [batch_size, 256, T]
        x = self.max_pool(x)
        x = self.dropout(x)
        
        x = self.conv4(x)     # [batch_size, 384, T]
        x = self.max_pool(x)
        x = self.dropout(x)
        
        x = x.mean(-1)        # [batch_size, 384]
        x = torch.cat([x, x_last], dim=1)  # [batch_size, 384 + input_size]
        x = self.dropout(x)
        x = self.fc1(x)       # [batch_size, 64]
        x = self.linear(x)    # [batch_size, 1]
        return x


#class ResBlock(nn.Module):
#    def __init__(self, input_size, output_size, kernel_cnn, pad, drop):
#        super(ResBlock, self).__init__()
#        self.conv1 = nn.Conv1d(input_size, output_size, kernel_cnn, padding = pad)
#        self.bn1 = nn.BatchNorm1d(output_size)
#        self.relu = nn.ReLU()
#        self.conv2 = nn.Conv1d(output_size, output_size, kernel_cnn, padding = pad)
#        self.bn2 = nn.BatchNorm1d(output_size)
#        self.drop = nn.Dropout(drop)
#
#        self.downsample = nn.Sequential(
#            nn.Conv1d(input_size, output_size, kernel_size=1, padding=0),
#            nn.BatchNorm1d(output_size),
#            nn.ReLU()
#        )
#
#    def forward(self, x):
#        identity = self.downsample(x)
#            
#        out = self.conv1(x)
#        out = self.bn1(out)
#        out = self.relu(out)
#        out = self.conv2(out)
#        out = self.bn2(out)
#
#        if out.size() != identity.size():
#            diff = out.size(2) - identity.size(2)
#            if diff > 0:
#                identity = nn.functional.pad(identity, (diff // 2, diff - diff//2))
#            elif diff < 0:
#                out = nn.functional.pad(out, (-diff//2, -diff + diff//2))
#                
#        out += identity
#        out = self.relu(out)
#        return self.drop(out)
#    

#class CNN(nn.Module):
#    def __init__(self, input_size, kernel_cnn, drop, delay):
#        super(CNN, self).__init__()
#        padd = (kernel_cnn // 2) + 1 if (kernel_cnn // 2) % 2 == 0 else (kernel_cnn // 2)
#        
#        self.features = nn.Sequential(
#            nn.Conv1d(input_size, 64, kernel_size = kernel_cnn, padding = padd),
#
#            ResBlock(input_size = 64, output_size = 32, kernel_cnn = kernel_cnn, pad = padd, drop = drop),
#            nn.ReLU(),
#
#            ResBlock(input_size = 32, output_size = 32, kernel_cnn = kernel_cnn, pad = padd, drop = drop),
#            nn.ReLU()
#
#        )
#
#        self.adaptative_pool = nn.AdaptiveAvgPool1d(delay)
#
#        flat_size = 32 * delay
#        self.fc = nn.Sequential(
#            nn.Linear(flat_size, 32),
#            nn.BatchNorm1d(32),
#            nn.ReLU(),
#            nn.Dropout(drop),
#
#            nn.Linear(32, 16),
#            nn.BatchNorm1d(16),
#            nn.ReLU(),
#            
#            nn.Linear(16, 1)
#        )
#    
#        self.apply(self._init_weights)
#
#    def _init_weights(self, m):
#        if isinstance(m, nn.Conv1d):
#            nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
#        elif isinstance(m, nn.Linear):
#            nn.init.xavier_normal_(m.weight)
#            nn.init.constant_(m.bias, 0)
#
#
#    def forward(self, x):
#        batch_size = x.size(0)
#        x = self.features(x)
#        x = self.adaptative_pool(x)
#        x = x.view(batch_size, -1)
#        x = self.fc(x)
#        return x

        

#* LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, drop, delay, num_layers_lstm, n_neurons = 128):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, n_neurons, num_layers_lstm, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(n_neurons * 2)
        self.fc = nn.Sequential(
            nn.Linear(n_neurons * 2, 64),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.layer_norm(out)
        return self.fc(out)
    


