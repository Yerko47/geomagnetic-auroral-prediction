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

from sklearn.metrics import root_mean_squared_error, r2_score, d2_absolute_error_score, d2_tweedie_score
from models import ANN, CNN, LSTM


#* Selection Model
def type_nn(type_model, x_train, drop, kernel_cnn, num_layer_lstm, delay, device) :
    """
    Selects and instantiates the specified neural network model, configuring it for the execution device (Cuda/CPU/GPU).

    Args
        type_model : str
            Type of model to create (e.g., 'ANN', 'CNN', 'LSTM').
        x_train : np.ndarray
            Training data used to infer the input dimensions.
        drop : float
            Dropout probability to prevent overfitting.
        kernel_cnn : int
            Kernel size for convolutional layers (only for CNN models).
        delay : int
            Time sequence length (only for LSTM models).
        num_layer_lstm : int
            Number of recurrent layers in the LSTM model.
        device : torch.device
            Device where the model will be hosted ('cuda', 'mps', 'cpu').

    Returns
        model : torch.nn.Module
            Instantiated model moved to the specified device.
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
    Computes regression metrics to compare actual vs. predicted values, 
    handling PyTorch tensors and avoiding device-related issues.

    Args
        real : torch.Tensor
            Tensor containing the real/target values.
        pred : torch.Tensor
            Tensor containing the predicted values from the model.

    Returns
        rmse : float
            Root Mean Squared Error (RMSE) between the actual and predicted values.
        r : float
            Coefficient of determination (R²) between the actual and predicted values, 
            ensured to be >= 0.
        d2_absolute_loss : float
            D² absolute error score between the actual and predicted values, 
            ensured to be >= 0.
        d2_tweedie_loss : float
            D² Tweedie score between the actual and predicted values, 
            ensured to be >= 0.
    """

    real_np = real.detach().cpu().numpy()
    pred_np = pred.detach().cpu().numpy()

    rmse = root_mean_squared_error(real_np, pred_np)
    d2_absolute_loss = d2_absolute_error_score(real_np, pred_np)
    d2_absolute_loss = d2_absolute_loss if d2_absolute_loss >= 0 else 0
    d2_tweedie_loss = d2_tweedie_score(real_np, pred_np)
    d2_tweedie_loss = d2_tweedie_loss if d2_tweedie_loss >= 0 else 0

    r2 = r2_score(real_np, pred_np)
    r = np.sqrt(abs(r2)) if r2 >= 0 else 0

    return rmse, r, d2_absolute_loss, d2_tweedie_loss


#* Seed
def set_seed(s) :
    """
    Fixes all random seeds to ensure that results can be replicated in future runs.

    Args
        s : int
            Seed value to set for all random number generators (e.g., NumPy, PyTorch, random).
    """
    torch.manual_seed(s)
    np.random.seed(s)


#* Train Model
def train_model(model, criterion, optimizer, train_loader, val_loader, EPOCH, lr, delay, type_model, auroral_index, schler, patience_schler, device, model_file = None, seed = 42):
    """
    Trains a PyTorch model with validation evaluation, saves the best model, and tracks metrics.

    Args
        model : nn.Module
            PyTorch model to train.
        criterion : torch.nn.Module
            Loss function to optimize.
        optimizer : str
            Optimizer to use (e.g., 'SGD', 'Adam').
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        EPOCH : int
            Total number of training epochs.
        lr : float
            Initial learning rate.
        delay : int
            Time delay used in the data processing.
        type_model : str
            Architecture type of the model (e.g., 'CNN', 'LSTM').
        auroral_index : str
            Target auroral index to predict.
        schler : str
            Learning rate scheduler to use (e.g., 'StepLR', 'ReduceLROnPlateau').
        patience_schler : int
            Number of epochs with no improvement before the scheduler reduces the learning rate.
        device : str
            Device to use for training ('cuda' or 'cpu').
        model_file : str
            Base path to save the trained model.
        seed : int
            Seed value for random number generation (default: 42).

    Returns
        model : nn.Module
            Trained model with the best weights.
        metrics_df : pd.DataFrame
            DataFrame containing the training and validation metrics history.
    """
    set_seed(seed)      # Set random seed for reproducibility

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
        'train_rmse': [], 'train_r_score': [], 'train_d2_abs': [], 'train_d2_tweedie': [],
        'val_rmse': [], 'val_r_score': [], 'val_d2_abs': [], 'val_d2_tweedie': [],
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
            

        for metric, value in zip(['rmse', 'r_score', 'd2_abs', 'd2_tweedie'], train_metrics) :        # Update metrics history
            metrics_history [f'train_{metric}'].append(value)
        
        for metric, value in zip(['rmse', 'r_score', 'd2_abs', 'd2_tweedie'], val_metrics) :        # Update metrics history
            metrics_history [f'val_{metric}'].append(value)

            
        if epoch % 10 == 0 or epoch == EPOCH - 1:
            header = f"\n{'='*40} Epoch {epoch + 1:03d}/{EPOCH} {'='*40}\n"

            train_log = (
                f"🔵 TRAIN: "
                f"RMSE: {train_metrics[0]:.4f} | "
                f"R: {train_metrics[1]:.4f} | "
                f"D² abs: {train_metrics[2]:.4f} | "
                f"D² tweedie: {train_metrics[3]:.4f}"
            )
            val_log = (
                f"🟠 VALID: "
                f"RMSE: {val_metrics[0]:.4f} | "
                f"R: {val_metrics[1]:.4f} | "
                f"D² abs: {val_metrics[2]:.4f} | "
                f"D² tweedie: {val_metrics[3]:.4f}"
            )
            print(header)
            print(train_log)
            print(val_log)

                    
    model.load_state_dict(best_model_wts)        # Load the best weights into the final model            

    metrics_df = pd.DataFrame({        # Create DataFrame with Metrics
        **{f'Train_{k.capitalize()}_{delay}': v for k, v in metrics_history.items() if 'train' in k},
        **{f'Valid_{k.capitalize()}_{delay}': v for k, v in metrics_history.items() if 'val' in k}
    })

    return model, metrics_df


#* Test Model
def model_testing(model, criterion, test_loader, model_file, type_model, auroral_index, delay, test_epoch, device) :
    """
    Evaluates a neural network model using a test dataset.

    Args
        model : nn.Module
            PyTorch model to be evaluated.
        criterion : torch.nn.Module
            Loss function used to compute the error.
        test_loader : torch.utils.data.DataLoader
            DataLoader for the test dataset.
        model_file : str
            Base path to save the evaluated model (if necessary).
        type_model : str
            Architecture type of the model (e.g., 'CNN', 'LSTM').
        auroral_index : str
            Target auroral index to predict.
        delay : int
            Time delay used in the data processing.
        test_epoch : pd.DataFrame
            Time data from the test dataset.
        device : str
            Device to use for evaluation ('cuda' or 'cpu').

    Returns
        result_df : pd.DataFrame
            DataFrame containing the results of the evaluation.
        metrics_df : pd.DataFrame
            DataFrame containing the evaluation metrics for the test dataset.
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

    rmse, r, d2_abs, d2_tweedie = metrics(real_tensor, pred_tensor)

    print(f'\nTest RMSE: {rmse:.4f}, Test R: {r:.4f}, Test D² abs: {d2_abs:.4f}, Test D² tweedie: {d2_tweedie:.4f}\n')
    
    metrics_df = pd.DataFrame({
        'Test_rmse': [rmse],
        'Test_r_score': [r],
        'Test_d2_abs': [d2_abs],
        'Test_d2_tweedie': [d2_tweedie]
    })

    result_df = pd.DataFrame({
        'Epoch': test_epoch,
        'Real': real_tensor.numpy().flatten().tolist(),
        'Pred': pred_tensor.numpy().flatten().tolist()
    })

    if auroral_index == 'AL_INDEX':
        result_df['Real'] = -1 * result_df['Real']
        result_df['Pred'] = -1 * result_df['Pred']
    
    return result_df, metrics_df

