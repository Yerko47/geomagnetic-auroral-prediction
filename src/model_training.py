"""
Code for selection and training model        # 
"""

import os
import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import root_mean_squared_error, r2_score
from src.models import ANN, CNN, LSTM


#* Selection Model
def type_nn(type_model, x_train, drop, kernel_cnn, num_layer_lstm, delay, device) :
    """
    Selects and instantiates the specified neural network model, configuring for the execution device (Cuda/CPU/GPU).

    Args:
        - type_model (str) : Type of model to create
        - x_train (np.array) : Training data for dimension inference
        - drop (float) : Dropout probability
        - kernel_cnn (int) : Kernel size (only CNN)
        - delay (int) : Time Squance length (Only LSTM)
        - num_layer_lstm (int) : Number of recurrent layers (Only LSTM)
        - device (torch.device) : Device where the model will be hosted ('cuda', 'mps', 'cpu').

    Return:
        - model (torch.nn.Module) : Model instantiated and moved to the specified device.

    """

    match type_model:
        case 'ANN':        # Multilayer Perceptron Model
            input_size = x_train.shape[1]
            model = ANN(input_size, drop).to(device)

        case 'CNN':        # Convolutional Model
            input_size = x_train.shape[1]
            model = CNN(input_size, kernel_cnn, drop).to(device)

        case 'LSTM':        # Recurrent Model
            input_size = x_train.shape[2]
            model = LSTM(input_size, drop, delay, num_layer_lstm).to(device)
        
        case _: raise ValueError("Invalid type_model specified")
    
    return model


#* Metrics
def metrics(real, pred) :
    """
    Computes regression metrics to compare actual vs. predicted values, handling PyTorch tensors and avoiding device issues.

    Args:
        - real (torch.Tensor) : Tensor with the real/target values
        - pred (torch.Tensor) : Tensor with the model predictions

    Return:
        - rmse (float) : Root Mean Squared Error
        - r (float) : Coefficient of determination (ensured >= 0)

    """

    real_np = real.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    rmse = root_mean_squared_error(real_np, pred_np)
    r2 = r2_score(real_np, pred_np)
    r = np.sqrt(abs(r2)) if r2 >= 0 else 0

    return rmse, r


#* Seed
def set_seed(s) :
    """
    Fix all random seeds so that results can be replicated in future runs.

    Args:
        - s (int) : Seed value

    """
    torch.manual_seed(s)
    np.random.seed(s)


#* Train Model
def train_model(model, criterion, optimizer, train_loader, val_loader, EPOCH,
                lr, delay, type_model, auroral_index, schler, patience_schler, 
                device, model_file = None) :
    """
    Training the PyTorch model with validation evaluation, saving the best model, and tracking metrics.

    Args:
        - model (nn.Module) : Pytorch model to train
        - criterion (Loss Function) : Loss function
        - optimizer (str) : Select optimizer
        - train_loader (DataLoader) : DataLoader for training data
        - val_loader (DataLoader) : DataLoader for validation data
        - EPOCH (int) : Total number of training epochs
        - lr (float) : Initial learning rate
        - delay (int) : Time delay used in the data
        - type_model (str) : Architecture type
        - auroral_index (str) : Target auroral index
        - schler (str) : Select Scheduler
        - patience_schler (int) : Patience Scheduler
        - device (str) : Device to use ('cuda' or 'cpu')
        - model_file (str) : Base path to save the model

    Return:
        - model (nn.Module) : Trained model with the best weights.
        - metrics_df (pd.DataFrame) : DataFrame with training and validation metrics history.
        
"""

    best_val_loss = float('inf')        # Initialize with infinite value
    best_model_wts = deepcopy(model.state_dict())        # Initial backup

    match optimizer:        # Optimizer selection
        case 'Adam': optimizer = optim.Adam(model.parameters(), lr = lr)
        case 'SGD': optimizer = optim.SGD(model.parameters(), lr = lr)
        case _: ValueError("Invalid optimizer specified")
    
    scheduler = None
    if schler:        # Scheduler selection
        match schler:
            case 'Reduce':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min', patience=patience_schler)
            case 'Cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, EPOCH * len(train_loader))
            case _:
                raise ValueError("Invalid scheduler specified")


    metrics_history = {
        'train_rmse': [], 'train_r': [], 
        'val_rmse': [], 'val_r': []
                       }
        
    for epoch in range(EPOCH) :
        if epoch == 0:
            print(f'Starting Cycle')

        model.train()         
        all_real_train, all_pred_train = [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(x) 
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()

            if scheduler: 
                if schler == 'Cosine': scheduler.step()


            all_real_train.append(y.cpu())
            all_pred_train.append(yhat.cpu())
        
        train_metrics = metrics(torch.cat(all_real_train), torch.cat(all_pred_train))

            
        model.eval()
        val_loss = 0.0
        all_real_val, all_pred_val = [], []
        with torch.no_grad() :        # Validation Phase
            for x, y in val_loader:
                yhat = model(x)
                loss = criterion(yhat, y)
                val_loss += (loss.item() * x.size(0))

                all_real_val.append(y.cpu())
                all_pred_val.append(yhat.cpu())
        
        val_loss /= len(val_loader.dataset)
        val_metrics = metrics(torch.cat(all_real_val), torch.cat(all_pred_val))

        if scheduler and schler =='Reduce':        # Update scheduler
            scheduler.step(val_loss)

            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
            if model_file:
                torch.save(best_model_wts, 
                           os.path.join(model_file, f"{type_model}_{auroral_index}_delay_{delay}.pt")
                           )
            

        for metric, value in zip(['rmse', 'r'], train_metrics) :        # Update metrics history
            metrics_history [f'train_{metric}'].append(value)
        
        for metric, value in zip(['rmse', 'r'], val_metrics) :        # Update metrics history
            metrics_history [f'val_{metric}'].append(value)

            
        if epoch % 10 == 0 or epoch == EPOCH - 1:
            log_str = (
             f"Epoch {epoch + 1:03d}/{EPOCH} | "
             f"Train RMSE: {train_metrics[0]:.4f} | Train R: {train_metrics[1]:.4f} | "
             f"Val RMSE: {val_metrics[0]:.4f} | Val R: {val_metrics[1]:.4f}"
            )
            print(log_str)

                    
    model.load_state_dict(best_model_wts)        # Load the best weights into the final model            

    metrics_df = pd.DataFrame({        # Create DataFrame with Metrics
        **{f'Train_{k.capitalize()}_{delay}': v for k, v in metrics_history.items() if 'train' in k},
        **{f'Valid_{k.capitalize()}_{delay}': v for k, v in metrics_history.items() if 'val' in k}
    })

    return model, metrics_df


#* Test Model
def model_testing(model, criterion, test_loader, model_file, type_model, auroral_index, delay, test_epoch, device) :
    """
    Function to evaluate a neural network model using a test dataset.

    Args:
        - model (nn.Module) : PyTorch model to be evaluated
        - criterion (loss Function) : Loss Function
        - test_loader (DataLoader) : DataLoader for test data
        - model_file (str) : Base path to save the model
        - type_model (str) : Architecture type
        - auroral_index (str) : Target auroral index
        - delay (int) : Time delay used in the data
        - test_epoch (pd.DataFrane) : Time data from test data
        - device (str) : Device to use ('cuda' or 'cpu')

    Return:
        - result_df (pd.DataFrame) : 
        - metrics_df (pd.DataFrame) :

    """
    
    model.load_state_dict(torch.load(
        os.path.join(model_file, f"{type_model}_{auroral_index}_delay_{delay}.pt"),
                     map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    
    model = model.to(device)
    model.eval()

    all_real, all_pred = [], []

    with torch.no_grad() :
        for x, y in test_loader:
            yhat = model(x)
            loss = criterion(yhat, y)

            all_real.append(y.cpu())
            all_pred.append(yhat.cpu())

    real_tensor = torch.cat(all_real)
    pred_tensor = torch.cat(all_pred)

    rmse, r = metrics(real_tensor, pred_tensor)

    print(f'Test RMSE: {rmse:.4f}, Test R: {r:.4f}')
    
    metrics_df = pd.DataFrame({
        'Test_RMSE': [rmse],
        'Test_R': [r]
    })

    result_df = pd.DataFrame({
        'Epoch': test_epoch,
        'Real': real_tensor.numpy().flatten().tolist(),
        'Pred': pred_tensor.numpy().flatten().tolist()
    })
    
    return result_df, metrics_df

