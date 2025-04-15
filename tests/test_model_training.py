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

    return temp_model_folder + '/'





#* Create model file directory
@pytest.fixture(scope="session")
def model_file_dir(test_folders):
    """
    Create a temporary directory to store model files during testing
    """
    model_dir = os.path.join(test_folders['temp'], "models")
    os.makedirs(model_dir, exist_ok=True)
    
    return model_dir + "/"


#* Test DataLoaders
@pytest.fixture
def data_loaders(processed_data_testing):
    """
    Create DataLoaders for testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get processed data
    x_train, y_train = processed_data_testing['x_train'], processed_data_testing['y_train']
    x_val, y_val = processed_data_testing['x_val'], processed_data_testing['y_val']
    x_test, y_test = processed_data_testing['x_test'], processed_data_testing['y_test']
    test_epoch = processed_data_testing['test_epoch']
    
    # Create datasets
    train_dataset = DataTorch(x_train, y_train, device)
    val_dataset = DataTorch(x_val, y_val, device)
    test_dataset = DataTorch(x_test, y_test, device)
    
    # Create dataloaders with smaller batch sizes for testing
    test_batch_train = min(batch_train, 32, len(train_dataset))
    test_batch_val = min(batch_val, 32, len(val_dataset))
    test_batch_test = min(batch_test, 32, len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=test_batch_train, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_test, shuffle=False)
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'test_epoch': test_epoch,
        'device': device
    }


#* Test Model Selection and Creation
def test_type_nn(processed_data_testing, data_loaders):
    """
    Test model selection and instantiation functionality
    """
    device = data_loaders['device']
    x_train = processed_data_testing['x_train']
    delay = processed_data_testing['delay']
    
    # Test ANN creation
    ann_model = type_nn('ANN', x_train, drop, kernel_cnn, num_layer_lstm, delay, device)
    assert isinstance(ann_model, ANN), "ANN model should be instantiated"
    assert next(ann_model.parameters()).device.type == device.type, f"ANN Model should be on {device.type}"
    
    # Test CNN creation
    cnn_model = type_nn('CNN', x_train, drop, kernel_cnn, num_layer_lstm, delay, device)
    assert isinstance(cnn_model, CNN), "CNN model should be instantiated"
    assert next(cnn_model.parameters()).device.type == device.type, f"CNN Model should be on {device.type}"
    
    # Test LSTM creation
    lstm_model = type_nn('LSTM', x_train, drop, kernel_cnn, num_layer_lstm, delay, device)
    assert isinstance(lstm_model, LSTM), "LSTM model should be instantiated"
    assert next(lstm_model.parameters()).device.type == device.type, f"LSTM Model should be on {device.type}"
    
    # Test invalid model type
    with pytest.raises(ValueError):
        type_nn('INVALID_MODEL', x_train, drop, kernel_cnn, num_layer_lstm, delay, device)


#* Test Metrics Function
def test_metrics():
    """
    Test metrics calculation functionality
    """
    # Create test data
    real = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    pred = torch.tensor([[1.1], [2.2], [2.7], [4.2]], dtype=torch.float32)
    
    # Calculate metrics
    rmse, r = metrics(real, pred)
    
    # Check metrics (giving some tolerance for floating point comparison)
    assert isinstance(rmse, float), "RMSE should be a float"
    assert isinstance(r, float), "R should be a float"
    assert 0.0 <= r <= 1.0, "R should be between 0.0 and 1.0"
    assert 0.2 <= rmse <= 0.3, f"RMSE should be around 0.25, got {rmse}"
    
    # Test with perfect prediction
    rmse_perfect, r_perfect = metrics(real, real)
    assert rmse_perfect == 0.0, "RMSE should be 0.0 for perfect prediction"
    assert r_perfect == 1.0, "R should be 1.0 for perfect prediction"


#* Test Seed Function
def test_set_seed():
    """
    Test seed setting functionality for reproducible results
    """
    # Test two models created with same seed produce the same weights
    seed = 42
    
    # First model with seed 42
    set_seed(seed)
    model1 = nn.Linear(10, 1)
    
    # Second model with seed 42
    set_seed(seed)
    model2 = nn.Linear(10, 1)
    
    # Third model with different seed
    set_seed(seed + 1)
    model3 = nn.Linear(10, 1)
    
    # Compare weights - models with same seed should have identical weights
    for p1, p2, p3 in zip(model1.parameters(), model2.parameters(), model3.parameters()):
        assert torch.all(torch.eq(p1, p2)), "Models with same seed should have identical weights"
        assert not torch.all(torch.eq(p1, p3)), "Models with different seeds should have different weights"


#* Test Model Forward Pass
def test_model_forward_pass(processed_data_testing, data_loaders):
    """
    Test forward pass for each model type
    """
    device = data_loaders['device']
    delay = processed_data_testing['delay']
    
    # Get a batch from data loader
    train_loader = data_loaders['train_loader']
    x_batch, y_batch = next(iter(train_loader))
    
    # Test ANN forward pass
    ann_model = type_nn('ANN', processed_data_testing['x_train'], drop, kernel_cnn, num_layer_lstm, delay, device)
    ann_output = ann_model(x_batch)
    assert ann_output.shape == y_batch.shape, f"ANN output shape {ann_output.shape} should match target shape {y_batch.shape}"
    
    # Adjust for CNN input structure if needed (it expects different dimensions than ANN)
    if type_model == 'CNN':
        cnn_model = type_nn('CNN', processed_data_testing['x_train'], drop, kernel_cnn, num_layer_lstm, delay, device)
        cnn_output = cnn_model(x_batch)
        assert cnn_output.shape == y_batch.shape, f"CNN output shape {cnn_output.shape} should match target shape {y_batch.shape}"
    
    # Test LSTM forward pass
    lstm_model = type_nn('LSTM', processed_data_testing['x_train'], drop, kernel_cnn, num_layer_lstm, delay, device)
    lstm_output = lstm_model(x_batch)
    assert lstm_output.shape == y_batch.shape, f"LSTM output shape {lstm_output.shape} should match target shape {y_batch.shape}"


#* Test Model Training (shorter version for testing)
def test_train_model(processed_data_testing, data_loaders, model_file_dir):
    """
    Test model training functionality with a small number of epochs
    """
    device = data_loaders['device']
    x_train = processed_data_testing['x_train']
    delay = processed_data_testing['delay']
    
    # Create model for training test
    test_model_type = type_model  # Using the selected model type from variables
    model = type_nn(test_model_type, x_train, drop, kernel_cnn, num_layer_lstm, delay, device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Set very small number of epochs for testing
    test_epochs = 2
    
    # Train model
    trained_model, metrics_df = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=data_loaders['train_loader'],
        val_loader=data_loaders['val_loader'],
        EPOCH=test_epochs,
        lr=lr,
        delay=delay,
        type_model=test_model_type,
        auroral_index=auroral_index,
        schler=schler,
        patience_schler=patience_schler,
        device=device,
        model_file=model_file_dir
    )
    
    # Check if model file was created
    model_path = f"{model_file_dir}{test_model_type}_{auroral_index}_delay_{delay}.pt"
    assert os.path.exists(model_path), f"Model file should be created at {model_path}"
    
    # Check returned model
    assert isinstance(trained_model, nn.Module), "Trained model should be a torch Module"
    
    # Check metrics DataFrame
    assert isinstance(metrics_df, pd.DataFrame), "Metrics should be returned as DataFrame"
    expected_cols = [f'Train_Rmse_{delay}', f'Train_R_{delay}', 
                     f'Valid_Rmse_{delay}', f'Valid_R_{delay}']
    for col in expected_cols:
        assert col in metrics_df.columns, f"Metrics DataFrame should have column {col}"
    assert len(metrics_df) == test_epochs, f"Metrics DataFrame should have {test_epochs} rows"


#* Test Model Testing Function
def test_test_model(processed_data_testing, data_loaders, model_file_dir):
    """
    Test model evaluation functionality
    """
    # Train a model first to have a model file to test
    test_train_model(processed_data_testing, data_loaders, model_file_dir)
    
    device = data_loaders['device']
    x_train = processed_data_testing['x_train']
    delay = processed_data_testing['delay']
    test_epoch = data_loaders['test_epoch']
    
    # Create model for testing
    test_model_type = type_model
    model = type_nn(test_model_type, x_train, drop, kernel_cnn, num_layer_lstm, delay, device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Evaluate model
    result_df, metrics_df = test_model(
        model=model,
        criterion=criterion,
        test_loader=data_loaders['test_loader'],
        model_file=model_file_dir,
        type_model=test_model_type,
        auroral_index=auroral_index,
        delay=delay,
        test_epoch=test_epoch,
        device=device
    )
    
    # Check results
    assert isinstance(result_df, pd.DataFrame), "Results should be returned as DataFrame"
    assert 'Epoch' in result_df.columns, "Results should include Epoch column"
    assert 'Real' in result_df.columns, "Results should include Real values"
    assert 'Pred' in result_df.columns, "Results should include Predicted values"
    
    # Check metrics
    assert isinstance(metrics_df, pd.DataFrame), "Test metrics should be returned as DataFrame"
    assert 'Test_RMSE' in metrics_df.columns, "Test metrics should include RMSE"
    assert 'Test_R' in metrics_df.columns, "Test metrics should include R"


#* Test All Models with Full Pipeline
@pytest.mark.parametrize("model_type", ["ANN", "CNN", "LSTM"])
def test_full_pipeline(model_type, processed_data_testing, data_loaders, model_file_dir):
    """
    Test the full pipeline for all model types
    """
    # Set global variable temporarily
    old_type_model = globals()['type_model']
    globals()['type_model'] = model_type
    
    try:
        device = data_loaders['device']
        x_train = processed_data_testing['x_train']
        delay = processed_data_testing['delay']
        test_epoch = data_loaders['test_epoch']
        
        # Create model
        model = type_nn(model_type, x_train, drop, kernel_cnn, num_layer_lstm, delay, device)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Train model for only 1 epoch to keep test fast
        test_epochs = 1
        
        # Train model
        trained_model, train_metrics_df = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=data_loaders['train_loader'],
            val_loader=data_loaders['val_loader'],
            EPOCH=test_epochs,
            lr=lr,
            delay=delay,
            type_model=model_type,
            auroral_index=auroral_index,
            schler=schler,
            patience_schler=patience_schler,
            device=device,
            model_file=model_file_dir
        )
        
        # Test model
        result_df, test_metrics_df = test_model(
            model=model,
            criterion=criterion,
            test_loader=data_loaders['test_loader'],
            model_file=model_file_dir,
            type_model=model_type,
            auroral_index=auroral_index,
            delay=delay,
            test_epoch=test_epoch,
            device=device
        )
        
        # Basic checks that everything works
        assert isinstance(train_metrics_df, pd.DataFrame), f"{model_type}: Training metrics should be a DataFrame"
        assert isinstance(test_metrics_df, pd.DataFrame), f"{model_type}: Test metrics should be a DataFrame"
        assert isinstance(result_df, pd.DataFrame), f"{model_type}: Results should be a DataFrame"
        
    finally:
        # Restore original type_model
        globals()['type_model'] = old_type_model


if __name__ == "__main__":
    pytest.main(["-v"])