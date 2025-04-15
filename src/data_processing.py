"""
Code for reading, preprocessing and processing data
"""
import os
import cdflib
import pandas as pd
import numpy as np
from datetime import timedelta

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

#* READ CDF
def cdf_read(file):
    """
    Loads CDF (Common Data Format) files and converts them to a DataFrame.
    This function opens a CDF file, extracts all variables, converts to time in the 'Epoch' column to a readable format, and rename some columns to improve the clarity.
    The result is a DataFrame ready for analysis.

    Args:
        - fike (str): Path of the CDF file you want to load.

    Return:
        - cdf_df (pd.Dataframe): A DataFrame containing the data from the CDF file.
        
    """

    cdf = cdflib.CDF(file)        # Open CDF file

    cdf_dict = {}
    info = cdf.cdf_info()        # Info about the CDF file is obtained (including variables)

    for key in info.zVariables:        # zVariables it is a list of the variables present in the CDF file
        cdf_dict[key]  = cdf[key]        # The corresponding data is extracted from the CDF file 

    cdf_df = pd.DataFrame(cdf_dict)

    if 'Epoch' in cdf_df.columns:
        cdf_df['Epoch'] = pd.to_datetime(cdflib.cdfepoch.encode(cdf_df['Epoch'].values))        # Converts the Epoch column time from CDF format to readable format (datetime64[ns])

    cdf_df.rename(columns = {'E': 'E_Field', 'F': 'B_Total'}, inplace = True)

    return cdf_df


#* DATA CLEANING
def bad_data(cdf_df):
    """
    Replaces erroneous values (maximum values) in each column of the DataFrame with NaN.
    This function iterates over the columns of the DataFrame, excluding the temporary column. To do this, it calculates the maximum value, rounds it to two decimal places, and replaces all values greater than or equal to this value with NaN and interpolated.

    Args:
        - df (pd.DataFrame): The DataFrame containing the data to be processed.

    Return:
        - processed_df (pd.DataFrame): The modified DataFrame with erroneous values replaced by the interpolation

    """

    processed_df = cdf_df.copy()

    for cols in processed_df.columns:
        if cols == 'Epoch' or cdf_df[cols].dtype == 'datetime64[ns]':        # Ignores temporary values
            continue

        max_value = int(processed_df[cols].max() * 100) / 100        # Calculate the maximum value and round it to two decimal places
        processed_df.loc[processed_df[cols] >= max_value, cols] = np.nan        # Replace values greater than or equal to the maximum value with NaN

        processed_df[cols] = processed_df[cols].interpolate(method = 'linear')        # A linear interpolation is performed
        processed_df[cols] = processed_df[cols].bfill(limit = 3)        # Limited padding of residual NaNs, with a limit of 3 to minimize errors

    return processed_df


#* DATASET BUILDING
def dataset(in_year, out_year, omni_file, raw_file, processOMNI):
    """
    Code responsible for creating a database by applying previosly dfined function to process and clean data.

    Args:
        - in_year (int): The starting year 
        - out_year (int): The ending year
        - omni_file (str): Address where the omni data
        - raw_file (str): Address where the raw files are saved
        - processOMNI (bool): Check if you want to perform this process or not

    Return:
        - df_omni (pd.Pandas): DataFrame with all correct values

    """

    start_time = pd.Timestamp(in_year, 1 ,1)
    end_time = pd.Timestamp(out_year, 3, 1)        # Setup time range

    save_feather = raw_file + f'omni_data_{in_year}_to_{out_year}.feather'

    if not os.path.exists(save_feather):
        if processOMNI:
            print(f"\n Processing omni data. From {start_time} to {end_time.strftime('%Y%m%d 23:59:00')}")
            print('::::::::::::::::::::::::::::::::::::::::::::::::::::: \n')

            date_array = pd.date_range(start = start_time, end = end_time, freq = 'MS')        # Generate monthly dates (freq = 'MS' --> start of every month)
            
            data_set = []

            for date in date_array:
                name_file = omni_file + f"{date.year}/omni_hro_1min_{date.strftime('%Y%m%d')}_v01.cdf"
                cdf = cdf_read(name_file)        # Read CDF files
                print(f'The file {name_file} is loading')
                data_set.append(cdf)

            df_omni = pd.concat(data_set, axis = 0, ignore_index = True)        # Concatenate each month

            df_omni.index = df_omni.Epoch
            df_omni.drop(columns = ['YR', 'Day', 'HR', 'Minute', 'IMF', 'PLS', 'IMF_PTS', 'PLS_PTS',
                                 'percent_interp', 'Timeshift', 'RMS_Timeshift', 'RMS_phase', 'Time_btwn_obs', 
                                 'RMS_SD_B', 'RMS_SD_fld_vec','Mach_num', 'Mgs_mach_num', 'ASY_D', 'ASY_H', 
                                 'PC_N_INDEX','x','y','z'], inplace=True)
            
            df_omni = bad_data(df_omni)        # Data Cleaning

            df_omni.reset_index(drop = True).to_feather(save_feather)

            print(f' ¡¡ Lets Go !! the {len(df_omni)} data has been successfully saved to {save_feather}')

        return df_omni
    
    elif os.path.exists(save_feather):
        df_omni = pd.read_feather(save_feather)

        return df_omni
    
    else:
        raise FileNotFoundError(f'The file {save_feather} does not exist')
    

#* STORM SELECTION
def storm_selection(df, processed_file):
    """
    Extracts 48-hours time windows around storm events from a time series dataset for focused analysis.
    Processes OMNI-like space weather data by isolating critical periods around storm events, enabling detailed study of pre-storm, storm peak, and post-storm conditions. 

    Args:
        - df (pd.DataFrame): Input time serie data containing Epoch column (Timestamps in ISO/OMNI format)
        - processed_file (str): Path to the directory containing 'storm_list.csv'

    Return:
        - df_storm (pd.DataFrame): Storm data with continuous 48-hour segments for each storm.

    """
    
    df['Epoch'] = pd.to_datetime(df['Epoch'])
    df.set_index('Epoch', inplace=True)

    storm_list = pd.read_csv(
                os.path.join(processed_file, 'storm_list.csv'),
                header = None,
                names = ['Epoch'],
                parse_dates = ['Epoch']        # Convert directly to datetime
            )
    
    df_storm = pd.DataFrame()
    for storm_time in storm_list['Epoch']:
        start = storm_time - pd.Timedelta('24h')
        end = storm_time + pd.Timedelta('24h')
        storm_window = df.loc[start:end].copy()        # Extract data from the main DataFrame

        df_storm = pd.concat([df_storm, storm_window], axis=0)
    
    return df_storm.reset_index()


#* SCALER DATASET
def scaler_df(df, scaler_type, auroral_param, omni_param):
    """
    This function applies a specified scaling method to the solar parameters in the DataSet.

    Args:
        - df (pd.DataFrame): Dataset with storm selection
        - scaler_type (str): The scaling method to apply ('robust', 'standard', 'minmax')
        - auroral_param (str): Matrix containing the indices to be predicted
        - omni_param (str): Matrix containing the solar parameters to use

    Return:
        - df (pd.DataFrame): Dataset with scaling in solar parameters

    """

    df_epoch = df[['Epoch']]
    df_index = df[auroral_param + ['SYM_D', 'SYM_H']]
    df_solar = df[omni_param]       # Solar parameters to scale

    match scaler_type:
        case 'robust': scaler = RobustScaler()        # Robustness to outliers (using medians and quantiles)
        case 'standard': scaler = StandardScaler()        # Standardization (mean=0, std=1)
        case 'minmax': scaler = MinMaxScaler()        # Normalization to range [0,1]
        case _: raise ValueError('Scaler must be "robust", "standard" or "minmax" ')

    df_solar_scaler = scaler.fit_transform(df_solar)        # Apply transformation
    df_solar_scaler = pd.DataFrame(df_solar_scaler, columns = df_solar.columns, index = df_solar.index)
    
    df = pd.concat([df_epoch, df_solar_scaler, df_index], axis = 1)

    return df
    
        
#* SET PREDICTION
def create_set_prediction(df, set_split, test_size, val_size):
    """
    Splits a dataset into training, validation, and test sets for predictive modeling, supporting two splitting strategies: organized (temporal) and random.

    Args:
        - df (pd.DataFrame): Scaled and temporally ordered DataSet 
        - set_split (str): Split method ('organized' or 'random')
        - test_size (float): Test set proportion
        - val_size (float): Validation set proportion
    
    Return:
        - train_df (pd.DataFrame): Training data
        - val_df (pd.DataFrame): Validation data
        - test_df (pd.DataFrame): Testing data

    """

    match set_split:

        case 'organized':        # Sequential division
            n = len(df)
            test_index = int(n * (1 - test_size)); val_index = int(test_index * (1 - val_size))

            train_df = df[:val_index].copy()        # First data (train)
            val_df = df[val_index:test_index].copy()        # Second data (val)
            test_df = df[test_index:].copy()        # Third data (test)

            train_df.reset_index(inplace = True, drop = True)
            val_df.reset_index(inplace = True, drop = True)
            test_df.reset_index(inplace = True, drop = True)

        case 'random':        # Stratified random split
            train_val_df, test_df = train_test_split(df, test_size = test_size, shuffle = False)
            train_df, val_df = train_test_split(train_val_df, test_size = val_size, shuffle = True)

        case _: raise ValueError('Set_split must be "organized" or "random"')

    print('---------- [ Percentage of sets ]----------\n')
    print(f'Training: {round(len(train_df) / len(df), 2) * 100}%')
    print(f'Validation: {round(len(val_df) / len(df), 2) * 100}%')
    print(f'Testing: {round(len(test_df) / len(df), 2) * 100}%\n')

    return train_df, val_df, test_df
    

#* TIME DELAY
def time_delay(df, omni_param, auroral_index, delay, type_model, group):
    """
    Prepare temporal data for prediction models by applying delays (lag features) according to the model type.
    If the model is an ANN (Multilayer Perceptron) it will be [m, n*t], but if it is a recurrent or convolutional neural network it will be [m, [t,n]]

    Args:
        - df (pd.DataFrame): Input DataFrame with temporary columns.
        - omni_param (list): Solar parameters 
        - auroral_index (str): Auroral index to predict
        - delay (int): Number of time steps to consider (time window)
        - type_model (str): Model type
        - group (str): Type of data set to apply the delay (train, val, test)

    Return:
        - np_solar (np.array): Solar wind feature with delay applied
        - np_index (np.array): Auroral Index targets
        - df_epoch (pd.DataFrame): Only if group = 'test' time data is kept

    """

    df_solar = df[omni_param].copy()        # Targets
    df_index = df[auroral_index].copy().abs()        # Features
    np_index = df_index.to_numpy()

    if type_model == 'ANN':
        delay_columns = []
        for col in df_solar.columns:        # Delay-based lag generation (ANN)
            for lag in range(1, delay + 1):
                delay_col = df_solar[col].shift(lag).astype('float32')
                delay_col.name = f'{col}_{lag}'
                delay_columns.append(delay_col)

        df_solar = pd.concat([df_solar] + delay_columns, axis=1)
        df_solar.dropna(inplace=True)
        np_solar = df_solar.values
        np_index = np_index[delay:]

    elif type_model in ['LSTM', 'CNN']:
        sequence = [df_solar.iloc[i:i + delay].values 
                    for i in range(len(df_solar) - delay + 1)]        # Generation of time sequences
        
        np_solar = np.array(sequence, dtype=np.float32)        # Reduce memory use
        np_index = np_index[delay - 1:]        # Target alignment

        if type_model == 'CNN':
            np_solar = np.transpose(np_solar, (0, 2, 1))


    if group == 'test':
        df_epoch = df['Epoch'].iloc[delay - 1:].reset_index(drop=True)        # Time data
        return np_solar, np_index, df_epoch
    else:
        return np_solar, np_index
    

#* DATA TORCH 
class DataTorch(Dataset):
    """
    A PyTorch Dataset class for handling data with automatic device placement.
    Provides automatic conversion of tensors to appropriate data types, also has support for GPU/CPU devices and a standardized interface to PyTorch DataLoader.

    Args:
        - solar_wind (np.array): Solar wind parameters (features) in numpy array format
        - index (np.array): Auroral Index parameter (target) in numpy array format
        - device (str, torch.device): Target device ('cuda', 'mps', 'cpu') for automatic data placement

    Return:
        - x_torch (torch.Tensor): Features in torch tensor format
        - y_torch (torch.Tensor): Target in torch tensor format

    """
    def __init__(self, solar_wind, index, device):
        self.device = device       # Save device info

        self.x = solar_wind         # Convert and move features to device
        self.y = index        # Convert and move targets to device (ensuring the shape [n_samples, 1])

    def __len__(self) -> int:        # Return the number of samples in the dataset
        return(len(self.x))
    
    def __getitem__(self, idx : int) -> tuple [torch.Tensor, torch.Tensor]:        # Identify the value according to the index in the dimension
        x = torch.tensor(self.x[idx], dtype=torch.float32).to(self.device)
        y = torch.tensor(self.y[idx], dtype=torch.float32).unsqueeze(0).to(self.device)

        return x, y

