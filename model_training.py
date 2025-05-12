"""
Code for selection and training model        # 
"""

import os
import numpy as np
import pandas as pd
import random 
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
            model = CNN(input_size, kernel_cnn, drop, delay).to(device)

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
        real : numpy array
            Array containing the real/target values.
        pred : numpy array
            Array containing the predicted values from the model.

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

    rmse = root_mean_squared_error(real, pred)
    d2_absolute_loss = d2_absolute_error_score(real, pred)
    d2_absolute_loss = d2_absolute_loss if d2_absolute_loss >= 0 else 0
    d2_tweedie_loss = d2_tweedie_score(real, pred)
    d2_tweedie_loss = d2_tweedie_loss if d2_tweedie_loss >= 0 else 0

    r2 = r2_score(real, pred)
    r = np.sqrt(abs(r2)) if r2 >= 0 else 0

    return rmse, r, d2_absolute_loss, d2_tweedie_loss


#* Seed
def set_seed(seed) :
    """
    Fixes all random seeds to ensure that results can be replicated in future runs.

    Args
        seed : int
            Seed value to set for all random number generators (e.g., NumPy, PyTorch, random).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#* Early Stopping
class EarlyStopping:
    def __init__(self, patience, min_delta=0.0):
        """
        Early stopping to terminate training when validation loss does not improve.
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            
        

#* Train Model
def train_val_model(model, criterion, optimizer_type, train_loader, val_loader, EPOCH, lr, delay, type_model, auroral_index, schler, patience_schler, patience, device, model_file, seed = 42):
    """
  

    Returns
        model : nn.Module
            Model trained during the training and validation phase.
        metrics_df : pd.DataFrame
            DataFrame containing the evaluation metrics for the test dataset.
    """
    set_seed(seed)
    best_model_wts = deepcopy(model.state_dict())

    match optimizer_type:
        case 'Adam': 
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        case 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9, nesterov = True, weight_decay=1e-5)
        case _:
            raise ValueError(f"No Valid Optimizer name {optimizer_type}")

    if schler:
        match schler:
            case 'Reduce': 
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience_schler)
            case 'Cosine': 
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCH * len(train_loader))
            case 'CosineRW': 
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
            case _: 
                raise ValueError(f"No Valid Scheduler name {schler}")
    

    metrics_history = {
        'train_rmse': [], 'train_r_score': [], 'train_d2_abs': [], 'train_d2_tweddie': [],
        'val_rmse': [], 'val_r_score': [], 'val_d2_abs': [], 'val_d2_tweddie': [],
    }

    early_stopper = EarlyStopping(patience = patience)

    for epoch in range(EPOCH):
        if epoch == 0:
            print(f"{' '*5} Start Cyle -- Model {type_model} -- Delay {delay}")
        
        model.train()
        all_real_train, all_pred_train = [], []

        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.5)

            loss.backward()
            optimizer.step()

            all_pred_train.append(yhat.detach().squeeze(-1).cpu().numpy())
            all_real_train.append(y.detach().cpu().numpy())
        
        train_metrics = metrics(np.concatenate(all_real_train, axis = 0), np.concatenate(all_pred_train, axis = 0))

        model.eval()
        all_real_val, all_pred_val = [], []
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                yhat = model(x)
                loss = criterion(yhat, y)

                val_loss += (loss.item()) * x.size(0)

                all_pred_val.append(yhat.squeeze(-1).cpu().numpy())
                all_real_val.append(y.cpu().numpy())
        
        val_metrics = metrics(np.concatenate(all_real_val, axis = 0), np.concatenate(all_pred_val, axis = 0))
        val_loss /= len(val_loader.dataset)

        if schler == 'Reduce':
            scheduler.step(val_loss)
        else:
            scheduler.step()


        for metric, value in zip(['rmse', 'r_score', 'd2_abs', 'd2_tweddie'], train_metrics):
            metrics_history[f"train_{metric}"].append(value)
        
        for metric, value in zip(['rmse', 'r_score', 'd2_abs', 'd2_tweddie'], val_metrics):
            metrics_history[f"val_{metric}"].append(value)


        if epoch % 10 == 0 or epoch == EPOCH - 1:
            header = f"\n{'='*10} Época {epoch + 1:03d}/{EPOCH} {'='*10}\n"

            train_log = (
                f"--> Training: "
                f"RMSE: {train_metrics[0]:.4f} | "
                f"R: {train_metrics[1]:.4f} | "
                f"D² abs: {train_metrics[2]:.4f} | "
                f"D² tweedie: {train_metrics[3]:.4f}"
            )
            val_log = (
                f"--> Validation: "
                f"RMSE: {val_metrics[0]:.4f} | "
                f"R: {val_metrics[1]:.4f} | "
                f"D² abs: {val_metrics[2]:.4f} | "
                f"D² tweedie: {val_metrics[3]:.4f}"
            )
            print(header)
            print(train_log)
            print(val_log)

        early_stopper(val_loss)
        if early_stopper.early_stop:
            best_model_wts = deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(model_file, f"{type_model}_{auroral_index}_delay_{delay}.pt"))

    model.load_state_dict(best_model_wts)

    metrics_df = pd.DataFrame({
        **{f'{k.capitalize()}_{delay}': v for k, v in metrics_history.items() if 'train' in k},
        **{f'{k.capitalize()}_{delay}': v for k, v in metrics_history.items() if 'val' in k}
    })

    return model, metrics_df


#*Training Model
def model_testing(model, criterion, test_loader, model_file, type_model, auroral_index, test_epoch, device):
    """
    
    """
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    all_real, all_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            loss = criterion(yhat, y)

            all_pred.append(yhat.detach().squeeze(-1).cpu().numpy())
            all_real.append(y.detach().cpu().numpy())
        
        test_metrics = metrics(np.concatenate(all_real, axis = 0), np.concatenate(all_pred, axis = 0))

        print(f'\nTest RMSE: {test_metrics[0]:.4f}, Test R: {test_metrics[1]:.4f}, Test D² abs: {test_metrics[2]:.4f}, Test D² tweedie: {test_metrics[3]:.4f}\n')

    metrics_df = pd.DataFrame({
        'Test_rmse': [test_metrics[0]],
        'Test_r_score': [test_metrics[1]],
        'Test_d2_abs': [test_metrics[2]],
        'Test_d2_tweddie': [test_metrics[3]]
    })

    result_df = pd.DataFrame({
        'Epoch': test_epoch,
        'Real': np.concatenate(all_real, axis = 0).tolist(),  # Valores reales
        'Pred': np.concatenate(all_pred, axis = 0).tolist()   # Valores predichos
    })

    if auroral_index == 'AL_INDEX':
        result_df['Real'] = -1 * result_df['Real']
        result_df['Pred'] = -1 * result_df['Pred']
    
    return result_df, metrics_df