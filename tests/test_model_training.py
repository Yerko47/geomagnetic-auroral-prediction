"""
PyTest script for model_training.py and models.py functionality        # 
"""

import os
import pytest
import pandas as pd
import numpy as np 
import tempfile
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from test_data_processing import *

import sys
from pathlib import Path

# Make sure the project root directory is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.variables import *
from src.data_processing import DataTorch
from src.models import ANN, CNN, LSTM
from src.model_training import *

#* Test Folders
@pytest.fixture(scope="session")
def model_folder(test_folders):
    """
    Create a temporary directory to store model files during testing
    """
    temp_model_folder = os.path.join(test_folders['temp'], 'models')
    os.makedirs(temp_model_folder, exist_ok = True)

    return temp_model_folder              


#* Test DataLoaders
@pytest.fixture
def DataTorch_testing(processed_data_testing):
    """
    Create DataLoaders for testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_set = [batch_train, batch_val, batch_test]
    data = {}

    models = ['ANN', 'CNN', 'LSTM']
    set_names = ['train', 'val', 'test']

    for model in models:
        for i, set_name in enumerate(set_names):
            data_torch = DataTorch(processed_data_testing[f'x_{set_name}_{model}'], processed_data_testing[f'y_{set_name}_{model}'], device)

            shuffle_set = True if set_name == 'train' else False
            data_loader = DataLoader(data_torch, batch_size = batch_set[i], shuffle = shuffle_set)

            data[f'{set_name}_loader_{model}'] = data_loader

    
    data['device'] = device

    return data


#* Check Metrics Function
def test_metrics():
    """
    Check metrics calculation function
    """
    real = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    pred = torch.tensor([[1.1], [2.2], [2.7], [4.2]], dtype=torch.float32)
    
    rmse, r, d2_abs, d2_tweedie = metrics(real, pred)

    assert isinstance(rmse,(float, int, np.float32)), "RMSE should be a float"
    assert isinstance(r, (float, int, np.float32)), "R should be a float"
    assert isinstance(d2_abs, (float, int, np.float32)), "D² Abs should be a float"
    assert isinstance(d2_tweedie, (float, int, np.float32)), "D² Tweedie should be a float"
    assert 0.0 <= r <= 1.0, "R should be between 0.0 and 1.0"
    assert d2_abs <= 1.0, "D² Abs should be less than 1.0"
    assert d2_tweedie <= 1.0, "D² Tweedie should be less than 1.0"
    assert 0.0 <= rmse, f"RMSE should be around 0.25, got {rmse}"


#* Check Seed Function
def test_set_seed():
    """
    Test seed setting functionality for reproducible results
    """
    seed = 42

    set_seed(seed)
    model_1 = nn.Linear(5, 1)

    set_seed(seed)
    model_2 = nn.Linear(5, 1)

    set_seed(seed + 1)
    model_3 = nn.Linear(5, 1)

    for p1, p2, p3 in zip(model_1.parameters(), model_2.parameters(), model_3.parameters()):
        assert torch.all(torch.eq(p1, p2)), "Models with same seed should have identical weights"
        assert not torch.all(torch.eq(p1, p3)), "Models with different seeds should have different weights"


#* Check Model Selection
def test_type_nn(processed_data_testing, DataTorch_testing):
    """
    Check model selection and instantiation functionality and test forward pass for each model type
    """
    device = DataTorch_testing['device']
    delay = processed_data_testing['delay']

    models_type = ['ANN', 'LSTM', 'CNN']

    for model_name in models_type:
        x_train = processed_data_testing[f'x_train_{model_name}']
        
        model = type_nn(model_name, x_train, drop, kernel_cnn, num_layer_lstm, delay, device)

        for x_batch, y_batch in DataTorch_testing[f'train_loader_{model_name}']:
            output = model(x_batch)
            break

        if model_name == 'ANN':
            assert isinstance(model, ANN), "ANN model should be instantiated"
            assert next(model.parameters()).device.type == device.type, f"ANN Model should be on {device.type}"
            assert output.shape == y_batch.shape, f"ANN output shape {output.shape} should match target shape {y_batch.shape}"

        elif model_name == 'CNN':
            assert isinstance(model, CNN), "CNN model should be instantiated"
            assert next(model.parameters()).device.type == device.type, f"CNN Model should be on {device.type}"
            assert output.shape == y_batch.shape, f"CNN output shape {output.shape} should match target shape {y_batch.shape}"
        
        elif model_name == 'LSTM':
            assert isinstance(model, LSTM), "LSTM model should be instantiated"
            assert next(model.parameters()).device.type == device.type, f"LSTM Model should be on {device.type}"
            assert output.shape == y_batch.shape, f"LSTM output shape {output.shape} should match target shape {y_batch.shape}"
        


#* Check Training Model
def test_train_model(processed_data_testing, DataTorch_testing, model_folder):
    """
    Test model training functionality with a small number od epochs
    """

    device = DataTorch_testing['device']
    delay = processed_data_testing['delay']

    test_epoch = 2
    models_type = ['ANN', 'CNN', 'LSTM']
    criterion = nn.MSELoss()

    for model_name in models_type:
        x_train = processed_data_testing[f'x_train_{model_name}']
        model = type_nn(model_name, x_train, drop, kernel_cnn, num_layer_lstm, delay, device)

        trained_model, metrics_df = train_model(
            model = model,
            criterion = criterion,
            optimizer = optimizer_type,
            train_loader = DataTorch_testing[f'train_loader_{model_name}'],
            val_loader = DataTorch_testing[f'val_loader_{model_name}'],
            EPOCH = test_epoch,
            lr = lr,
            delay = delay,
            type_model = model_name,
            auroral_index = auroral_index,
            schler = schler,
            patience_schler = patience_schler,
            device = device,
            model_file = model_folder
        )

        model_path = f"{model_folder}\\{model_name}_{auroral_index}_delay_{delay}.pt"
        assert os.path.exists(model_path), f"Model file should be created at {model_path}"

        assert isinstance(trained_model, nn.Module), "Trained model should be a torch Module"

        assert isinstance(metrics_df, pd.DataFrame), "Metrics should be returned as DataFrame"
        
        assert len(metrics_df) == test_epoch, f"Metrics DataFrame should have {test_epoch} rows"


#* Check Testing model
def test_model_testing(processed_data_testing, DataTorch_testing, model_folder):
    """
    Test model evaluation functionality
    """
    test_train_model(processed_data_testing, DataTorch_testing, model_folder)

    device = DataTorch_testing['device']
    delay = processed_data_testing['delay']

    models_type = ['ANN', 'CNN', 'LSTM']
    criterion = nn.MSELoss()

    for model_name in models_type:
        x_train = processed_data_testing[f'x_train_{model_name}']
        test_epoch = processed_data_testing[f'test_epoch_{model_name}']
        model = type_nn(model_name, x_train, drop, kernel_cnn, num_layer_lstm, delay, device)

        result_df, metrics_df = model_testing(
            model = model,
            criterion = criterion,
            test_loader = DataTorch_testing[f'test_loader_{model_name}'],
            model_file = model_folder,
            type_model = model_name,
            auroral_index = auroral_index,
            delay = delay,
            test_epoch = test_epoch,
            device = device
        )

        assert isinstance(result_df, pd.DataFrame), "Results should be returned as DataFrame"
        assert 'Epoch' in result_df.columns, "Results should include Epoch column"
        assert 'Real' in result_df.columns, "Results should include Real values"
        assert 'Pred' in result_df.columns, "Results should include Predicted values"

        assert isinstance(metrics_df, pd.DataFrame), "Test metrics should be returned as DataFrame"
        assert 'Test_RMSE' in metrics_df.columns, "Test metrics should include RMSE"
        assert 'Test_R' in metrics_df.columns, "Test metrics should include R"

        if auroral_index == 'AL_INDEX':
            assert (result_df['Real'].values < 1).all() and (result_df['Pred'].values < 1).all(), "AL_INDEX values should be less than 1"
        else:
            assert (result_df['Real'].values > 1).all() and (result_df['Pred'].values > 1).all(), "AE_INDEX values should be greater than 1"



if __name__ == "__main__":
    pytest.main(["-v"])