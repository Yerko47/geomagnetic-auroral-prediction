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
    Loads CDF (Common Data Format) files and converts them into a DataFrame.

    This function opens a CDF file, extracts all variables, converts the 'Epoch' column to a readable time format, 
    and renames some columns for clarity. The result is a DataFrame ready for analysis.

    Args
        file : str
            Path to the CDF file to be loaded.

    Returns
        cdf_df : pd.DataFrame
            DataFrame containing the data from the CDF file.
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
    Replaces erroneous maximum values in each column of the DataFrame with NaN and interpolates the result.

    This function iterates over the columns of the DataFrame (excluding the temporary column), 
    computes the maximum value, rounds it to two decimal places, and replaces all values greater than 
    or equal to this maximum with NaN. The resulting missing values are then interpolated.

    Args
        df : pd.DataFrame
            DataFrame containing the data to be processed.

    Returns
        processed_df : pd.DataFrame
            Modified DataFrame with erroneous values replaced by interpolated values.
    """

    processed_df = cdf_df.copy()

    for cols in processed_df.columns:
        if cols == 'Epoch' or cdf_df[cols].dtype == 'datetime64[ns]':        # Ignores temporary values
            continue

        max_value = np.floor(processed_df[cols].max() * 100) / 100        # Calculate the maximum value and round it to two decimal places
        processed_df.loc[processed_df[cols] >= max_value, cols] = np.nan        # Replace values greater than or equal to the maximum value with NaN

        processed_df[cols] = processed_df[cols].interpolate(method = 'linear')        # A linear interpolation is performed
        processed_df[cols] = processed_df[cols].bfill(limit = 3)        # Limited padding of residual NaNs, with a limit of 3 to minimize errors

    return processed_df


#* DATASET BUILDING
def dataset(in_year, out_year, omni_file, raw_file, processOMNI):
    """
    Creates a database by applying previously defined functions to process and clean space weather data.

    Args
        in_year : int
            The starting year for data processing.
        out_year : int
            The ending year for data processing.
        omni_file : str
            Path to the directory containing OMNI data files.
        raw_file : str
            Path to the directory containing raw data files.
        processOMNI : bool
            Whether to process the OMNI data (True) or skip this step (False).

    Returns
        df_omni : pd.DataFrame
            Cleaned and processed DataFrame containing the relevant OMNI data.
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
            df_omni.drop(columns = ['YR', 'Day', 'HR', 'Minute', 'PLS', 'PLS_PTS',
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
    Extracts 48-hour time windows around storm events from a time series dataset for focused analysis.
    Processes OMNI-like space weather data by isolating critical periods around storm events, 
    enabling detailed study of pre-storm, storm peak, and post-storm conditions.

    Args
        df : pd.DataFrame
            Input time series data containing an 'Epoch' column with timestamps in ISO/OMNI format.
        processed_file : str
            Path to the directory containing the 'storm_list.csv' file.

    Returns
        df_storm : pd.DataFrame
            Subset of the dataset containing continuous 48-hour segments for each storm event.
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
    Scales the solar parameters in the dataset using the specified scaling method.
    The scaling methods available are 'robust', 'standard', and 'minmax'.
    The function also handles the inversion of the auroral index (AL_INDEX) to ensure that
    negative values are represented as positive in the scaled dataset.


    Args
        df : pd.DataFrame
            Dataset with storm selection.
        scaler_type : str
            The scaling method to apply. Options are 'robust', 'standard', or 'minmax'.
        auroral_param : str
            Name of the column or key representing the indices to be predicted.
        omni_param : str
            Name of the column or key representing the solar parameters to scale.

    Returns
        df : pd.DataFrame
            Dataset with the solar parameters scaled according to the selected method.
    """


    df_epoch = df[['Epoch']]
    df_index = df[auroral_param]
    df_solar = df[omni_param]      

    for cols in df_index.columns:
        if cols == 'AL_INDEX':
            df_index[cols] = -1 * df_index[cols]   
        break     
        

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
    Splits a dataset into training, validation, and test sets for predictive modeling. 
    Supports two splitting strategies: organized (temporal) and random.

    Args
        df : pd.DataFrame
            Scaled and temporally ordered dataset.
        set_split : str
            Split method to use. Options are 'organized' (temporal order) or 'random' (shuffled).
        test_size : float
            Proportion of the dataset to allocate to the test set.
        val_size : float
            Proportion of the dataset to allocate to the validation set.

    Returns
        train_df : pd.DataFrame
            Training data.
        val_df : pd.DataFrame
            Validation data.
        test_df : pd.DataFrame
            Testing data.
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

    return train_df, val_df, test_df
    

#* TIME DELAY
def time_delay(df, omni_param, auroral_index, delay, type_model, group):
    """
    Prepare temporal data for prediction models by applying delays (lag features) according to the model type.
    If the model is an ANN (Multilayer Perceptron), it will be shaped as [m, n*t], 
    but if it is a recurrent or convolutional neural network, it will be shaped as [m, t, n].
    
    Args
        df : pd.DataFrame
            Input DataFrame with temporary columns.
        omni_param : list
            Solar parameters.
        auroral_index : str
            Auroral index to predict.
        delay : int
            Number of time steps to consider (time window).
        type_model : str
            Model type.
        group : str
            Type of data set to apply the delay (train, val, test).
    
    Returns
        np_solar : np.array
            Solar wind features with delay applied.
        np_index : np.array
            Auroral index targets.
        df_epoch : pd.DataFrame
            Only returned if group='test'. Contains time data.
    """
    df_solar = df[omni_param].copy()        # Targets
    df_index = df[auroral_index].copy().abs()        # Features
    np_index = df_index.to_numpy()
    df_epoch = df['Epoch']

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

        if group == 'test':
            df_epoch = df_epoch.iloc[delay:].reset_index(drop=True).copy()


    elif type_model == 'LSTM':
        sequence = [df_solar.iloc[i : i + delay].values for i in range(len(df_solar) - delay + 1)]        # Generation of time sequences
        np_solar = np.array(sequence, dtype=np.float32)        # Reduce memory use
        np_index = np_index[delay - 1:]        # Target alignment

        if group == 'test':
            df_epoch = df['Epoch'].iloc[delay - 1:].reset_index(drop=True).copy() 


    elif type_model == 'CNN':
        sequence = [df_solar.iloc[i : i + delay].values for i in range(len(df_solar) - delay + 1)]
        np_solar = np.array(sequence, dtype = np.float32)
        np_solar = np.transpose(np_solar, (0, 2, 1))
        np_index = np_index[delay - 1:]


        if group == 'test':
            df_epoch = df['Epoch'].iloc[delay - 1:].reset_index(drop=True).copy()        # Time data

    if group == 'test':
        return np_solar, np_index, df_epoch
    else:
        return np_solar, np_index
    

#* DATA TORCH 
class DataTorch(Dataset):
    """
    A PyTorch Dataset class for handling data with automatic device placement.
    Provides automatic conversion of tensors to appropriate data types, supports GPU/CPU devices, 
    and offers a standardized interface for PyTorch DataLoader.

    Args
        solar_wind : np.ndarray
            Solar wind parameters (features) in NumPy array format.
        index : np.ndarray
            Auroral Index parameter (target) in NumPy array format.
        device : str or torch.device
            Target device ('cuda', 'mps', 'cpu') for automatic data placement.

    Returns
        x_torch : torch.Tensor
            Features converted to torch tensor format and moved to the specified device.
        y_torch : torch.Tensor
            Targets converted to torch tensor format and moved to the specified device.
    """

    def __init__(self, solar_wind, index, device):
        
        self.device = device       # Save device info

        self.x = torch.tensor(solar_wind, dtype=torch.float32).to(self.device)         # Convert and move features to device
        self.y = torch.tensor(index, dtype=torch.float32).unsqueeze(1).to(self.device)        # Convert and move targets to device (ensuring the shape [n_samples, 1])

    def __len__(self) -> int:        # Return the number of samples in the dataset
        return(len(self.x))
    
    def __getitem__(self, idx):        # Identify the value according to the index in the dimension
        x = self.x[idx]
        y = self.y[idx]

        return x, y

