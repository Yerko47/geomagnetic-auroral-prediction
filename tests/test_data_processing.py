"""
PyTest script for data_processing.py functionality        # 
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path

# Make sure the project root directory is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_processing import *
from src.variables import *


#* Test Folders
@pytest.fixture(scope="session")
def test_folders():
    """
    Create temporary folders for test data
    """

    temp_folder = tempfile.mkdtemp()
    raw_folder = os.path.join(temp_folder, "raw")
    processed_folder = os.path.join(temp_folder, "processed")

    os.makedirs(raw_folder, exist_ok = True)
    os.makedirs(processed_folder, exist_ok = True)

    yield {'temp': temp_folder, 'raw': raw_folder, 'processed': processed_folder}

    shutil.rmtree(temp_folder)


#* Artificial OMNI data
@pytest.fixture(scope = "session")
def artificial_omni_data(test_folders):
    """
    Create a artifcial OMNI dataset with semi realistic structure. This avoids using the original data loading function.
    """
    days = 60
    minutes = 24 * 60
    total_minutes = days * minutes        # Create a 60-day range

    start_time = pd.Timestamp(2010, 1, 1)
    dates = pd.date_range(start = start_time, end = start_time + timedelta(days), freq = '1min')        # Create a 60-days dataset with 1-min resolution
    dates = dates[:total_minutes]
    df = pd.DataFrame({'Epoch': dates})


    # Add OMNI parameters with realistic values and patterns
    for param in omni_param:
        match param:
            case "B_Total": df[param] = 5 + 2 * np.sin(np.arange(len(df)) / 1440) + 0.5 * np.random.randn(len(df))
            case "BX_GSE": df[param] = 3 * np.sin(np.arange(len(df)) / 960) + 0.4 * np.random.randn(len(df))
            case "BY_GSE": df[param] = 2 * np.cos(np.arange(len(df)) / 1200) + 0.3 * np.random.randn(len(df))
            case "BZ_GSE": df[param] = 2 * np.sin(np.arange(len(df)) / 1080) + 0.3 * np.random.randn(len(df))
            case "BY_GSM": df[param] = 2 * np.cos(np.arange(len(df)) / 1200 + 0.2) + 0.3 * np.random.randn(len(df))
            case "BZ_GSM": df[param] = 2 * np.sin(np.arange(len(df)) / 1080 + 0.5) + 0.3 * np.random.randn(len(df))
            case "flow_speed": df[param] = 400 + 50 * np.sin(np.arange(len(df)) / 4320) + 10 * np.random.randn(len(df))
            case "Vx": df[param] = -df['flow_speed'] * 0.9 + 10 * np.random.randn(len(df))
            case "Vy" | "Vz": df[param] = 20 * np.sin(np.arange(len(df)) / 2880) + 5 * np.random.randn(len(df))
            case "T": df[param] = 50000 + 20000 * (df['flow_speed'] - 400) / 100 + 5000 * np.random.randn(len(df))
            case "Pressure": df[param] = 1e-6 * (5 + np.random.randn(len(df))) * df['flow_speed']**2 + 0.5 * np.random.randn(len(df))
            case "E_Field": df[param] = abs(df['flow_speed'] * df['BZ_GSM'] * 1e-3) + 0.2 * np.random.randn(len(df))
    

    # Add auroral parameters with realistic correlations to solar wind
    for param in auroral_param:
        match param:
            case "AE_INDEX": df[param] = 100 + 50 * np.sin(2 * np.pi * (np.arange(len(df)) % 1440) / 1440) + 10 * abs(df['E_Field']) + 20 * np.maximum(0, -df['BZ_GSM']) + 20 * np.random.randn(len(df))
            case "AL_INDEX": df[param] = -0.8 * df['AE_INDEX'] + 10 * np.random.randn(len(df))
            case "AU_INDEX": df[param] = 0.3 * df['AE_INDEX'] + 15 * np.random.randn(len(df))


    # Add SYM indices
    df['SYM_D'] = 10 * np.sin(np.arange(len(df)) / 2880) + np.random.randn(len(df))
    df['SYM_H'] = -20 * np.sin(np.arange(len(df)) / 1440) + 2 * np.random.randn(len(df))


    # Create storm list for storm_selection() function
    storm_time = []
    unique_storm = []
    for i in range(5):
        rolling_max = df['AE_INDEX'].rolling(window = 360).max()
        storm_index = rolling_max.nlargest(5).index
        for idx in storm_index:
            if idx > 360 and idx < (len(df) - 360):
                storm_time.append(df['Epoch'].iloc[idx])

    if storm_time:
        storm_time = sorted(storm_time)
        unique_storm.append(storm_time[0])
        for t in storm_time[1:]:
            if (t - unique_storm[-1]).total_seconds() > 48 * 3600:        # Ensure at least 48 hours between storm times
                unique_storm.append(t)


    # Write storm list to file
    storm_df = pd.DataFrame({'Epoch': unique_storm})
    storm_df.to_csv(os.path.join(test_folders['processed'], 'storm_list.csv'), header=False, index=False)


    # Add bad data points to test bad_data function
    for param in df.columns:
        if param != "Epoch":
            bad_index = np.random.choice(range(len(df)), size = int(0.005 * len(df)), replace = False)
            max_value = df[param].max() * 1.5
            df.loc[bad_index, param] = max_value        # Add some maximum values that should be filtered by bad_data
    
    return df


#* Storm Selection Test
@pytest.fixture
def storm_selection_testing(artificial_omni_data, test_folders):
    """
    Fixture for storm selection data
    """
    df_copy = artificial_omni_data.copy()
    return storm_selection(df_copy, test_folders['processed'])


#* Bad data Test
@pytest.fixture
def bad_data_testing(storm_selection_testing):
    """
    Apply bad_data function for cleaning to storm data test
    """
    return bad_data(storm_selection_testing)


#* Scaler df Test
@pytest.fixture
def scaler_df_testing(bad_data_testing):
    """
    Fixture for scaler_df function
    """
    return scaler_df(bad_data_testing, scaler_type, auroral_param, omni_param)


#* Create set prediction Test
@pytest.fixture
def create_set_prediction_testing(scaler_df_testing):
    """
    Fixture for train/val/test split datasets
    """
    return create_set_prediction(scaler_df_testing, set_split, test_size, val_size)


#* Processed Data Test
@pytest.fixture
def processed_data_testing(create_set_prediction_testing):
    """
    Fixture for time-delayed data ready for model training
    """
    train_df, val_df, test_df = create_set_prediction_testing
    test_delay = 5
    models = ['ANN', 'CNN', 'LSTM']
    sets = {'train': train_df,
            'val': val_df,
            'test': test_df
            }
    data = {'delay': test_delay}

    for model in models:
        for set_name, df in sets.items():
            result = time_delay(df, omni_param, auroral_index, test_delay, model, set_name)

            if set_name == 'test':
                x, y, epoch = result
                data[f"x_{set_name}_{model}"] = x
                data[f"y_{set_name}_{model}"] = y
                data[f"test_epoch_{model}"] = epoch
            
            else:
                x, y = result
                data[f"x_{set_name}_{model}"] = x
                data[f"y_{set_name}_{model}"] = y

    return data
    



#* Check Artificial Data Structure
def test_artificial_data_structure(artificial_omni_data):
    """
    Ckeck data has expected structure
    """
    assert isinstance(artificial_omni_data, pd.DataFrame), "Synthetic data should be a pandas DataFrame"
    assert "Epoch" in artificial_omni_data.columns, "Data should have an Epoch column"
    assert artificial_omni_data["Epoch"].dtype == 'datetime64[ns]', "Epoch should be datetime type"

    for param in omni_param + auroral_param + ['SYM_D', 'SYM_H']:
        assert param in artificial_omni_data.columns, f"Data should contain {param} column"


#* Check Bad Data Cleaning
def test_bad_data(artificial_omni_data):
    """
    Check bad_data function
    """
    df_bad = artificial_omni_data.copy()

    test_col = omni_param[0]
    max_value = df_bad[test_col].max()
    extreme_index = np.random.choice(range(len(df_bad)), size = 10, replace = True)        # Intentionally add some extreme values to test cleaning
    df_bad.loc[extreme_index, test_col] = max_value * 2        # Clearly erroneous values

    df = bad_data(df_bad)
    
    assert df[test_col].max() < max_value * 2, f"Cleaning should have removed extreme values in {test_col}"
    assert not df[test_col].isna().any(), "Cleaning should have interpolated NaN values"

    for col in df.columns:
        if col != 'Epoch':
            assert df[col].max() <= df_bad[col].max(), f"Cleaning should not increase maximum value in {col}"
    

#* Check Storm Selection
def test_storm_selection(storm_selection_testing, artificial_omni_data):
    """
    Check Storm selection funcionality
    """
    assert isinstance(storm_selection_testing, pd.DataFrame), "Storm data should be a pandas DataFrame"
    assert 'Epoch' in storm_selection_testing.columns, "Storm data should have an Epoch column"
    
    assert len(storm_selection_testing) < len(artificial_omni_data), "Storm data should be a subset of the original data"


#* Check Scaler df
def test_scaler_df(scaler_df_testing, bad_data_testing):
    """
    Check data scaling functionality
    """
    assert isinstance(scaler_df_testing, pd.DataFrame), "Scaled data should be a pandas DataFrame"
    assert len(scaler_df_testing) == len(bad_data_testing), "Scaling should preserve the number of samples"

    for param in scaler_df_testing.columns:
        if param in omni_param:
            if scaler_type == 'robust':
                assert abs(scaler_df_testing[param].median()) < 0.5, f"{param} median should be close to 0 after robust scaling"
            elif scaler_type == 'standard':
                assert abs(scaler_df_testing[param].mean()) < 0.5, f"{param} mean should be close to 0 after standard scaling"
                assert 0.5 < scaler_df_testing[param].std() < 1.5, f"{param} std should be close to 1 after standard scaling"
            elif scaler_type == 'minmax':
                assert scaler_df_testing[param].min() >= -0.1, f"{param} min should be >= 0 after minmax scaling"
                assert scaler_df_testing[param].max() <= 1.1, f"{param} max should be <= 1 after minmax scaling"
        else:
            assert np.array_equal(scaler_df_testing[param].values, bad_data_testing[param].values), f"{param} should not be scaled"


#* Check Create set Prediction
def test_create_set_prediction(create_set_prediction_testing, scaler_df_testing):
    """
    Check create sets prediction function
    """

    train_df, val_df, test_df = create_set_prediction_testing

    total_samples = sum(len(df) for df in [train_df, val_df, test_df])
    assert total_samples == len(scaler_df_testing), f"Split should preserve total samples: {total_samples} vs {len(scaler_df_testing)}"

    expected_test_size = int(len(scaler_df_testing) * test_size)
    expected_val_size = int((len(scaler_df_testing) - expected_test_size) * val_size)
    expected_train_size = len(scaler_df_testing) - expected_test_size - expected_val_size
    
    assert abs(len(test_df) - expected_test_size) <= 1, f"Test split size incorrect: {len(test_df)} vs {expected_test_size}"
    assert abs(len(val_df) - expected_val_size) <= 1, f"Validation split size incorrect: {len(val_df)} vs {expected_val_size}"
    assert abs(len(train_df) - expected_train_size) <= 1, f"Training split size incorrect: {len(train_df)} vs {expected_train_size}"

    for df in [train_df, val_df, test_df]:
        assert set(df.columns) == set(scaler_df_testing.columns), "Column mismatch in split datasets"

    if set_split == 'organized':
        last_train_time = train_df['Epoch'].iloc[-1]
        first_val_time = val_df['Epoch'].iloc[0]
        last_val_time = val_df['Epoch'].iloc[-1]
        first_test_time = test_df['Epoch'].iloc[0]
        
        assert last_train_time <= first_val_time, "Training should end before validation starts in organized split"
        assert last_val_time <= first_test_time, "Validation should end before testing starts in organized split"


#* Check Time Delay
def test_time_delay(processed_data_testing):
    """
    Check time delay function
    """

    models = ['ANN', 'CNN', 'LSTM']

    for model in models:
        print(model)
        x_train, y_train = processed_data_testing[f'x_train_{model}'], processed_data_testing[f'y_train_{model}']
        x_val, y_val = processed_data_testing[f'x_val_{model}'], processed_data_testing[f'y_val_{model}']
        x_test, y_test = processed_data_testing[f'x_test_{model}'], processed_data_testing[f'y_test_{model}']
        test_epoch = processed_data_testing[f'test_epoch_{model}']
        test_delay = processed_data_testing['delay']
        print(x_train.shape)

        if model == 'ANN':
            expected_feature_dim = len(omni_param) * (test_delay + 1)
            assert x_train.shape[1] == expected_feature_dim, f"ANN feature dimension incorrect: {x_train.shape[1]} vs {expected_feature_dim}"
            assert x_train.ndim == 2, "ANN inputs should be 2D arrays"
        
        elif model == 'LSTM':
            assert x_train.ndim == 3, "LSTM inputs should be 3D arrays (batch, seq_len, features)"
            assert x_train.shape[2] == len(omni_param), f"LSTM feature dimension incorrect: {x_train.shape[2]} vs {len(omni_param)}"
            assert x_train.shape[1] == test_delay, f"LSTM sequence length incorrect: {x_train.shape[1]} vs {test_delay}"
        
        elif model == 'CNN':
            assert x_train.ndim == 3, "CNN inputs should be 3D arrays (batch, features, seq_len)"
            assert x_train.shape[1] == len(omni_param), f"CNN channel dimension incorrect: {x_train.shape[1]} vs {len(omni_param)}"
            assert x_train.shape[2] == test_delay, f"CNN sequence length incorrect: {x_train.shape[2]} vs {test_delay}"

        assert y_train.ndim <= 2, "Target should be 1D or 2D"
        if y_train.ndim == 2:
            assert y_train.shape[1] == 1, "Target should have shape (n_samples, 1) if 2D"

        assert len(x_train) == len(y_train), f"Train features and targets misaligned: {len(x_train)} vs {len(y_train)}"
        assert len(x_val) == len(y_val), f"Validation features and targets misaligned: {len(x_val)} vs {len(y_val)}"
        assert len(x_test) == len(y_test), f"Test features and targets misaligned: {len(x_test)} vs {len(y_test)}"
    
        assert len(test_epoch) == len(x_test), f"Test {model} epochs and features misaligned: {len(test_epoch)} vs {len(x_test)}"


#* Check DataTorch
def test_DataTorch(processed_data_testing):
    """
    Check DataTorch class and DataLoader functionality
    """
    models = ['ANN', 'CNN', 'LSTM']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for model in models:
        x_train, y_train = processed_data_testing[f'x_train_{model}'], processed_data_testing[f'y_train_{model}']

        train_dataset = DataTorch(x_train, y_train, device)
        assert len(train_dataset) == len(x_train), f"Dataset length incorrect: {len(train_dataset)} vs {len(x_train)}"

        x, y = train_dataset[0]
        assert isinstance(x, torch.Tensor), "Dataset should return torch.Tensor for features"
        assert isinstance(y, torch.Tensor), "Dataset should return torch.Tensor for targets"
        assert x.device.type == device, f"Tensor should be on {device}, but is on {x.device.type}"

        train_loader = DataLoader(train_dataset, batch_size = batch_train, shuffle = True)

        for batch_x, batch_y in train_loader:
            assert isinstance(batch_x, torch.Tensor), "DataLoader should yield torch.Tensor batches"
            assert batch_x.shape[0] <= batch_train, f"Batch size incorrect: {batch_x.shape[0]} vs {batch_train}"
            assert batch_x.dtype == torch.float32, f"Features should be float32, got {batch_x.dtype}"
            assert batch_y.dtype == torch.float32, f"Targets should be float32, got {batch_y.dtype}"
            break  


if __name__ == "__main__":
    pytest.main(["-v"])

