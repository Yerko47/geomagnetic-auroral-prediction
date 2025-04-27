import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

from sklearn.linear_model import LinearRegression


#* Setup Axis Style
def setup_axis_style(ax, xlabel = None, ylabel = None, xlabelsize = 15, ylabelsize = 15, ticksize = 15):
    """
    Configures common axis styling for a Matplotlib Axes object.

    This function allows customization of the axis labels, tick sizes, and grid settings. 
    It also applies a specific format to the y-axis labels if a y-axis label is provided.

    Args
    
    ax : matplotlib.axes.Axes
        The Axes object to apply the staling to.
    xlabel : str, optional
        The label for the x-axis (Default is None).
    ylabel : str, optional
        The label for the y-axis (Default is None).
    xlabelsize : int, optional
        Font size for the x-axis label (Default is 15).
    ylabelsize : int, optional
        Font size for the y-axis label (Default is 15).
    ticksize : int, optional
        Font size for the axis ticks (Default is 15).
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize = xlabelsize)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad = 30 ,fontsize = ylabelsize, va = 'center')
    
    ax.tick_params(axis = 'both', length =7, width = 2, colors = 'black', 
                   grid_color = 'black', grid_alpha = 0.4 ,
                   which = 'major', labelsize = ticksize)
    ax.grid(True)

    if ylabel:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_powerlimits((-3, 4))


#* Format Parameters Label
def format_parameter_label(param):
    """
    Standardizes the formatting of parameter labels for consistent display.

    This function applies specific formatting rules to various parameter names, such as adding units (e.g., [nT], [K]) 
    or converting names to a more readable or standardized form (e.g., replacing underscores with spaces).
    
    Args
        param : str
            The parameter name to format.

    Returns
        str
            The formatted parameter label with appropriate units or modified text.
    """
    if 'INDEX' in param: 
        return param.replace('_INDEX', '') + r' [nT]'
    elif 'beta' in param: 
        return r'$\beta$'
    elif 'B' in param: 
        if 'Total' in param : 
            return param.replace('_Total', ' T') + r' [nT]' 
        else: 
            return param.replace('_', ' ') + r' [nT]'   
    elif 'proton' in param: 
        return r'$\rho$' + r' [#N/cm$^3$]'
    elif 'V' in param:
        return param + r' [km/s]'
    elif 'T' in param and 'B' not in param:
        return param + r' [K]'
    elif 'E_Field' in param:
        return 'E' + r' [mV/m]'
    elif 'Pressure' in param:
        return 'P' + r' [nPa]'
    elif 'flow_speed' in param:
        return 'flow speed' + r' [km/s]'
    return param


#* Mapping Columns
def get_column_mapping(df):
    """
    Generates a dictionary to rename DataFrame columns into more readable or scientific formats.

    This function scans column names in a DataFrame and replaces specific substrings with 
    standardized or symbolic representations. It is useful for cleaning and formatting 
    column names for visualization or analysis.

    Parameters
    
        df : pd.DataFrame
            DataFrame whose column names will be processed.

    Returns
    -------
        dict
            A dictionary mapping original column names to formatted labels.
    """
    column_mapping = {}
    for cols in df.columns:
        if 'INDEX' in cols:
            column_mapping[cols] = cols.replace('_INDEX', '')
        if 'GSM' in cols:
            column_mapping[cols] = cols.title().replace('_Gsm', ' GSM')
        if 'GSE' in cols:
            column_mapping[cols] = cols.title().replace('_Gse', ' GSE')
        if 'Total' in cols:
            column_mapping[cols] = cols.replace('_Total', 't')
        if 'proton' in cols:
            column_mapping[cols] = r'$\rho$'
        if 'Pressure' in cols:
            column_mapping[cols] = 'P'
        if 'E_Field' in cols:
            column_mapping[cols] = 'E'
        if 'Beta' in cols:
            column_mapping[cols] = r'$\beta$'
        if 'flow' in cols:
            column_mapping[cols] = 'flow speed'
    return column_mapping
    

#* Load Storm Data
def load_storm_data(processed_file, min_date = None):
    """
    Loads storm event data from a CSV file and filters it by a minimum date if provided.

    This function reads a CSV file named 'storm_list.csv' from the specified directory, parses the 'Epoch' timestamps, 
    and optionally filters out storms that occurred before a given minimum date.

    Args:
        processed_file : str
            Path to the directory containing the storm list CSV file.
        min_date : str, optional
            Minimum date for filtering the storm data (Default is None).
    
    Returns:
        pd.DataFrame
            DataFrame containing the storm data, filtered by the specified minimum date.
    """
    storm_list = pd.read_csv(processed_file + 'storm_list.csv', header = None, names = ['Epoch'])
    storm_list['Epoch'] = pd.to_datetime(storm_list['Epoch'])

    if min_date:
        min_date = pd.to_datetime(min_date)
        storm_list = storm_list[storm_list['Epoch'] >= min_date]
    
    return storm_list


#* Time Serie Plot
def time_serie_plot(df, omni_param, auroral_param, processed_file, auroral_historic_file, solar_historic_file):
    """
    Generates time series graphs for solar and auroral parameters around identified geomagnetic storm events.

    Args:
        df : pd.DataFrame
            DataFrame containing the solar and auroral parameters.
        omni_param : list
            List of solar parameters to plot.
        auroral_param : list
            List of auroral parameters to plot.
        processed_file : str
            Path to the directory containing the storm list CSV file.
        auroral_historic_file : str
            Path to the directory where auroral plots will be saved.
        solar_historic_file : str
            Path to the directory where solar plots will be saved.
    """
    storm_list = load_storm_data(processed_file)
    df = df.set_index('Epoch')

    for storm_date in storm_list:
        start_time = storm_date - pd.Timedelta('24h')
        end_time = storm_date + pd.Timedelta('24h')

        period_data = df[(df.index >= start_time) & (df.index <= end_time)]

        fig_solar, axs_solar = plt.subplots(len(omni_param), 1, figsize = (12, 15), sharex = True, layout = 'constrained')
        fig_solar.suptitle(f'Solar Parameters from {start_time} to {end_time}', fontsize = 18, fontweight = 'bold')

        if len(omni_param) == 1:
            axs_solar = [axs_solar]
        
        for j, param in enumerate(omni_param):
            axs_solar[j].plot(period_data[param], color = 'teal', zorder = 1, linewidth = 1.5)
            setup_axis_style(axs_solar[j], ylabel = format_parameter_label(param))
            axs_solar[j].set_xlim(start_time, end_time)

        axs_solar[-1].set_xlabel('Date', fontsize = 15)

        fig_auroral, axs_auroral = plt.subplot(len(auroral_param), 1, figsize = (12, 6), sharex = True, layout = 'constrained')
        fig_auroral.suptitle(f'Auroral Parameters from {start_time} to {end_time}', fontsize = 18, fontweight = 'bold')

        if len(auroral_param) == 1:
            axs_auroral = [axs_auroral]

        for j, param in enumerate(auroral_param):
            axs_auroral[j].plot(period_data[param], color = 'teal', zorder = 1, linewidth = 1.5)
            setup_axis_style(axs_auroral[j], ylabel = format_parameter_label(param))
            axs_auroral[j].set_xlim(start_time, end_time)
        
        axs_auroral[-1].set_xlabel('Date', fontsize = 15)

        plt.subplots_adjust(left = 0.15)

        filename_solar = f'Omni_Parameters_{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}.png'
        filename_auroral = f'Auroral_electrojet_index_{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}.png'

        fig_solar.savefig(solar_historic_file + filename_solar)
        fig_auroral.savefig(auroral_historic_file + filename_auroral)
        plt.close(fig_solar)
        plt.close(fig_auroral)


#* Correlation Plot
def correlation_plot(df, corr_file):
    """
    Generates a correlation heatmap for the specified DataFrame and saves it as an image.

    Args:
        df : pd.DataFrame
            DataFrame containing the data for which the correlation matrix will be calculated.
        corr_file : str
            Path to the directory where the correlation heatmap image will be saved.    
    """

    correlation = ['pearson', 'kendall', 'spearman']

    for corr in correlation:
        matrix = round(df.corr(method = corr), 2)

        column_mapping = get_column_mapping(matrix)
        
        plt.figure(figsize = (14, 12))
        plt.title(f'{corr.upper()} Correlation Matrix', fontsize = 18, fontweight = 'bold')

        heatmap = plt.pcolor(matrix, cmap = 'BrBG', vmin = -1, vmax = 1, edgecolors = 'white', linewidth = 1)

        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                value = matrix.iloc[i, j]  

                color = 'white' if abs(value) >= 0.75 else 'black'
                plt.text(j + 0.5, i + 0.5, f'{value:.2f}', 
                         ha = 'center', va = 'center', color = color, 
                         fontsize = 11, fontweight = 'bold')

        cbar = plt.colorbar(heatmap)  
        cbar.set_label('Correlation', rotation = -90, labelpad = 25, fontsize = 15, fontweight = 'bold')
        cbar.ax.tick_params(labelsize = 15)

        matrix.rename(columns = column_mapping, index = column_mapping, inplace = True)

        plt.xticks(np.arange(0.5, len(matrix.columns), 1), matrix.columns,
                   fontsize = 12, rotation = 45, ha = 'right', fontweight = 'bold')
        plt.yticks(np.arange(0.5, len(matrix.index), 1), matrix.index,
                   fontsize = 12, fontweight = 'bold')
        
        plt.tight_layout()
        plt.savefig(corr_file + f'{corr}_correlation_matrix.png')
        plt.close()


#* Metrics Plot
def metrics_plot(metrics_train_val, delay, auroral_index, type_model, train_loss_file, train_r_score_file):
    """
    Generates and saves plots for RMSE and R-Squared metrics for training and validation datasets.

    Args:
        metrics_train_val : pd.DataFrame
            DataFrame containing the RMSE and R-Squared metrics for training and validation datasets.
        delay : int
            The delay parameter used in the model.
        auroral_index : str
            The name of the auroral index being predicted.
        type_model : str
            The type of model used for prediction.
        train_loss_file : str
            Path to the directory where RMSE plots will be saved.
        train_r_score_file : str
            Path to the directory where R-Squared plots will be saved.
    """
    metric_list = ['RMSE', 'R_Score']

    for metric in metric_list:
        plt.figure(figsize = (12, 10))
        title = f'{metric.replace('_', ' ').title()} for {auroral_index.replace('_INDEX', 'Index')} - {type_model} (Delay {delay})'
        plt.title(title, fontsize = 20, fontweight = 'bold')

        for cols in metrics_train_val.columns:
            if metric in cols and ('Train' in cols or 'Valid' in cols):
                metric_value = metrics_train_val[cols].values
                color = 'teal' if 'Train' in cols else 'red'
                label_name = 'Train' if 'Train' in cols else 'Valid'
                plt.plot(metric_value, color = color, label = label_name, linewidth = 2)
        
        setup_axis_style(plt.gca(), xlabel = 'Epoch', ylabel = metric.replace('_', ' ').title(), ticksize = 12)

        plt.legend(font_size = 15)
        filname = f'{metric.replace('_', ' ')}_Plot_Train/Valid_{auroral_index.replace("_INDEX", "Index")}_{type_model}_(Delay_{delay}).png'
        
        if 'RMSE' in metric:
            plt.savefig(train_loss_file + filname)
        else:
            plt.savefig(train_r_score_file + filname)
        plt.close()


#* Density Plot
def density_plot(metric_test, result_df, delay, test_density_file, auroral_index, type_model):
    """
    Generates and saves a density plot comparing the predicted and actual values of the auroral index.
    This function graphs the entire test set, NOT USING sotrm_list.csv 

    Args:
        metric_test : pd.DataFrame
            DataFrame containing the R-Squared metric for the test dataset.
        result_df : pd.DataFrame
            DataFrame containing the predicted and actual values of the auroral index.
        delay : int
            The delay parameter used in the model.
        test_density_file : str
            Path to the directory where density plots will be saved.
        auroral_index : str
            The name of the auroral index being predicted.
        type_model : str
            The type of model used for prediction.
    """
    r_score = metric_test['R_Score'].values[0]
    k = int(0.01 * np.sqrt(len(result_df)))

    if auroral_index == 'AL_INDEX':
        result_df['Real'] = result_df['Real'] * -1
        result_df['Pred'] = result_df['Pred'] * -1

    np_log_real = np.log10(result_df['Real'].values + 1e-10)
    np_log_pred = np.log10(result_df['Pred'].values + 1e-10)

    p_min = min(min(np_log_real), min(np_log_pred))
    p_max = max(max(np_log_real), max(np_log_pred))

    plt.figure(figsize = (15, 12))
    plt.title(f'Density Plot for {auroral_index.replace("_INDEX", "Index")} - {type_model} (Delay {delay})', fontsize = 20, fontweight = 'bold')

    hist = plt.hist2d(np_log_real, np_log_pred, norm = LogNorm())
    plt.plot([p_min, p_max], [p_min, p_max], color = 'black', linewidth = 2, label = f'R = {r_score:.2f}')

    cbar = plt.colorbar(hist[3])
    cbar.set_label('Density', rotation = -90, labelpad = 25, fontsize = 15, fontweight = 'bold')
    cbar.ax.tick_params(labelsize = 15)

    plt.xlabel('Log10(Real)', fontsize = 20)
    plt.ylabel('Log10(Pred)', fontsize = 20)

    plt.tick_params(axis = 'both', length = 7, width = 2, colors = 'black',
                    grid_color = 'black', grid_alpha = 0.4, which = 'major', labelsize = 15)
    
    plt.legend(loc = 'best', fontsize = 15)
    plt.grid(True)

    filname = f'Density_Plot_{auroral_index.replace("_INDEX", "Index")}_{type_model}_(Delay_{delay}).png'
    plt.savefig(test_density_file + filname)
    plt.close()


#* Delay Metrics Plot
def delay_metrics_plot(metric_test, delay_length, auroral_index, type_model, test_loss_file, test_r_score_file):
    """
    Generates and saves plots metrics for different delays.

    Args:
        metric_test : pd.DataFrame
            DataFrame containing the RMSE and R-Squared metrics for the test dataset.
        delay_length : list
            List of delay values used in the model.
        auroral_index : str
            The name of the auroral index being predicted.
        type_model : str
            The type of model used for prediction.
        test_loss_file : str
            Path to the directory where RMSE plots will be saved.
        test_r_score_file : str
            Path to the directory where R-Squared plots will be saved.
    """
    metrics = ['RMSE', 'R_Score']
    for metric in metrics:
        plt.figure(figsize=(10,6))
        plt.title(f'Test {metric.replace('_',' ')} for different delays ({auroral_index.replace("_INDEX", "Index")}) - {type_model}', fontsize = 20, fontweight = 'bold')

        metric_values = []
        for cols in metric_test.columns:
            if metric in cols:
                metric_values = metric_test[cols].values
                plt.plot(delay_length, metric_values, marker = 'o', color = 'teal', linewidth = 1.5, linestyle = 'dashed')
        
        setup_axis_style(plt.gca(), xlabel = 'Delay', ylabel = metric.replace('_', ' '), xlabelsize = 20, ylabelsize = 20, ticksize = 15)

        filname = f'{metric}_Compared_Test_{auroral_index.replace("_INDEX", "Index")}_{type_model}.png'
        
        if 'RMSE' in metric:
            plt.savefig(test_loss_file + filname)
        elif 'R_Score' in metric:
            plt.savefig(test_r_score_file + filname)

        plt.close()


#* Time Serie Test Plot    
def time_serie_test_plot(result_df, delay, auroral_index, type_model, test):
    """
    
    """
