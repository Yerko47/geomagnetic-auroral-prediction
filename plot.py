import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib import dates
from matplotlib.gridspec import GridSpec

import imageio.v2 as imageio
from io import BytesIO


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
        ax.set_ylabel(ylabel, labelpad = 30, fontsize = ylabelsize, va = 'center')
    
    ax.tick_params(axis = 'both', length = 7, width = 2, colors = 'black', 
                   grid_color = 'black', grid_alpha = 0.4 ,
                   which = 'major', labelsize = ticksize)
    ax.grid(True)

    if ylabel:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText = True))
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
    elif 'SYM' in param:
        return param.replace('_', ' ') + r'[nT]'
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
        return 'Q' + r' [km/s]'
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
        if 'SYM' in cols:
            column_mapping[cols] = cols.replace('_', '')
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

    for storm_date in storm_list['Epoch']:
        start_time = storm_date - pd.Timedelta('24h')
        end_time = storm_date + pd.Timedelta('24h')

        period_data = df[(df.index >= start_time) & (df.index <= end_time)]

        fig_solar, axs_solar = plt.subplots(len(omni_param), 1, figsize=(12, 15), sharex=True, layout='constrained')
        gs = fig_solar.add_gridspec(len(omni_param), hspace = 0)
        fig_solar.suptitle(f'Solar Parameters from {start_time} to {end_time}', fontsize=18, fontweight='bold')


        if len(omni_param) == 1:
            axs_solar = [axs_solar]
        
        for j, param in enumerate(omni_param):
            axs_solar[j].plot(period_data[param], color='teal', zorder=1, linewidth=1.5)
            setup_axis_style(axs_solar[j], ylabel=format_parameter_label(param))
            axs_solar[j].set_xlim(start_time, end_time)

        axs_solar[-1].set_xlabel('Date', fontsize=15)

        fig_auroral, axs_auroral = plt.subplots(len(auroral_param), 1, figsize=(12, 6), sharex=True, layout='constrained')
        fig_auroral.suptitle(f'Auroral Parameters from {start_time} to {end_time}', fontsize=18, fontweight='bold')

        if len(auroral_param) == 1:
            axs_auroral = [axs_auroral]

        for j, param in enumerate(auroral_param):
            axs_auroral[j].plot(period_data[param], color='teal', zorder=1, linewidth=1.5)
            setup_axis_style(axs_auroral[j], ylabel=format_parameter_label(param))
            axs_auroral[j].set_xlim(start_time, end_time)
        
        axs_auroral[-1].set_xlabel('Date', fontsize=15)

        plt.subplots_adjust(left=0.15)

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
def metrics_plot(metrics_train_val, delay, auroral_index, type_model, train_loss_file, train_r_score_file, train_d2_abs_file, train_d2_tweedie_file):
    """
    Generates and saves plots for RMSE, R-Squared, D² Absolute Error Score and D² Tweedie Score metrics for training and validation datasets.

    Args:
        metrics_train_val : pd.DataFrame
            DataFrame containing the RMSE, R-Squared, D² Absolute Error Score and D² Tweedie Score metrics for training and validation datasets.
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
        train_d2_abs_file : str
            Path to the directory where D² Absolute Error Score plots will be saved.
        train_d2_tweedie_file : str
            Path to the directory where D² Tweedie Score plots will be saved.
    """
    metric_list = ['rmse', 'r_score', 'd2_abs', 'd2_tweedie']

    for metric in metric_list:
        plt.figure(figsize=(12, 10))
        
        if 'd2' in metric:
            title = f'{metric.replace("d2_", "D² ").title()} for {auroral_index.replace("_INDEX", " Index")} - {type_model} (Delay {delay})'
        else:
            title = f'{metric.replace("_", " ").title()} for {auroral_index.replace("_INDEX", " Index")} - {type_model} (Delay {delay})'
        
        plt.title(title, fontsize=20, fontweight='bold')

        train_col = None
        valid_col = None
        
        for cols in metrics_train_val.columns:
            if metric in cols:
                if 'Train' in cols or 'train' in cols:
                    train_col = cols
                elif 'Valid' in cols or 'val' in cols:
                    valid_col = cols
        
        if train_col:
            train_values = metrics_train_val[train_col].values
            plt.plot(range(1, len(train_values) + 1), train_values, 
                    color='teal', label='Train', linewidth=2)
        
        if valid_col:
            valid_values = metrics_train_val[valid_col].values
            plt.plot(range(1, len(valid_values) + 1), valid_values, 
                    color='red', label='Valid', linewidth=2)
        
        y_name = metric.replace('d2_', 'D² ').title() if 'd2' in metric else metric.upper() if 'RMSE' in metric else metric.replace('_', ' ').title()

        setup_axis_style(plt.gca(), xlabel='Epoch', ylabel = y_name, ticksize=10)

        plt.legend(fontsize=15)
        
        filename = f'{metric}_Plot_Train_Valid_{auroral_index.replace("_INDEX", "")}_{type_model}_Delay_{delay}.png'
        
        if 'rmse' in metric:
            plt.savefig(train_loss_file + filename)
        elif 'r_score' in metric:
            plt.savefig(train_r_score_file + filename)
        elif 'd2_abs' in metric:
            plt.savefig(train_d2_abs_file + filename)
        elif 'd2_tweedie' in metric:
            plt.savefig(train_d2_tweedie_file + filename)
        
        plt.close()


#* Delay Metrics Plot
def delay_metrics_plot(metric_test, delay_length, auroral_index, type_model, test_loss_file, test_r_score_file, test_d2_abs_file, test_d2_tweedie_file):
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
        test_d2_abs_file : str
            Path to the directory where D² Absolute Error Score plots will be saved.
        test_d2_tweedie_file : str
            Path to the directory where D² Tweedie Score plots will be saved.
    """

    metrics = ['rmse', 'r_score', 'd2_abs', 'd2_tweedie']
    
    for metric in metrics:
        plt.figure(figsize=(10,6))
        
        if 'd2' in metric:
            title = f'{metric.replace("d2_", "D² ").title()} for different delays ({auroral_index.replace("_INDEX", " Index")}) - {type_model}'
        else:
            title = f'{metric.replace("_", " ").upper()} for different delays ({auroral_index.replace("_INDEX", " Index")}) - {type_model}'

        plt.title(title, fontsize=20, fontweight='bold')

        test_cols = [col for col in metric_test.columns if metric in col and 'Test' in col]
        
        
        if test_cols:
            plt.plot(delay_length, metric_test[test_cols].values.flatten(), 
                     marker='o', color='teal', linewidth=1.5, 
                     linestyle='dashed', markersize=8)
                    
        
        y_name = metric.replace('d2_', 'D² ').title() if 'd2' in metric else metric.upper() if 'rmse' in metric else metric.replace('_', ' ').title()

        setup_axis_style(plt.gca(), xlabel='Delay', ylabel=y_name, xlabelsize=15, ylabelsize=15, ticksize=10)

        filename = f'{metric}_Compared_Test_{auroral_index.replace("_INDEX", "")}_{type_model}.png'
        
        if 'rmse' in metric:
            plt.savefig(test_loss_file + filename)
        elif 'r_score' in metric:
            plt.savefig(test_r_score_file + filename)
        elif 'd2_abs' in metric:
            plt.savefig(test_d2_abs_file + filename)
        elif 'd2_tweedie' in metric:
            plt.savefig(test_d2_tweedie_file + filename)

        plt.close()


#* Density and History Plot
def comparison_plot(result_file, processed_file, comparison_file, auroral_index, delay_length):
    """
    Generates and saves density and history plots comparing the predicted models values and real values of the auroral index.

    Args:
        result_file : str
            Path to the directory containing the model results.
        processed_file : str
            Path to the directory containing the storm list CSV file.
        comparison_file : str
            Path to the directory where comparison plots will be saved.
        auroral_index : str
            The name of the auroral index being predicted.
        delay_length : list
            List of delay values used in the model.
    """

    storm_list = load_storm_data(processed_file)
    model_list = ['LSTM', 'CNN', 'ANN']

    k = int(2 * np.sqrt(len(storm_list)))

    for i, storm in enumerate(storm_list['Epoch']):
        start_time = storm - pd.Timedelta('24h')
        end_time = storm + pd.Timedelta('24h')

        for delay in delay_length:
            fig = plt.figure(figsize = (18,10))
            title = f'Prediction from {start_time.strftime("%Y-%m-%d")} to {end_time.strftime("%Y-%m-%d")} for the {auroral_index.replace("_INDEX", " Index")}'
            fig.suptitle(title, fontsize=15, fontweight='bold')
            fig.subplots_adjust(top=0.92)
            gs = GridSpec(2, 3, figure = fig, height_ratios = [1, 2])
            gs.update(wspace = 0.3, hspace = 0.3)

            model_data = {}

            for j, model in enumerate(model_list):
                result_file = f'{result_file}_results_delay_{delay}_{auroral_index}_{model}.feather'
                metric_file = f'{result_file}_metrics_delay_{delay}_{auroral_index}_{model}.csv'

                df = pd.read_feather(result_file)
                df_metric = pd.read_csv(metric_file)

                storm_df = df[(df['Epoch'] >= start_time) & (df['Epoch'] <= end_time)]
                model_data[model] = storm_df

                ax_hist = fig.add_subplot(gs[0, j])

                real = np.log(np.abs(model_data[model][f'{auroral_index}_real'].values))
                pred = np.log(np.abs(model_data[model][f'{auroral_index}_pred'].values))

                h = ax_hist.hist2d(real, pred, bins = k, cmap = 'viridis', norm = LogNorm())

                p_min = min(min(real), min(pred))
                p_max = max(max(real), max(pred))   
                ax_hist.plot([p_min, p_max], [p_min, p_max], color = 'black', linewidth = 2)

                r = df_metric['R_Score'].values[0]

                ax_hist.set_title(f'{model} - R = {r:.2f}', fontsize = 15, fontweight = 'bold')
                setup_axis_style(ax_hist, xlabel = 'Log(Real)', ylabel = 'Log(Pred)', xlabelsize = 15, ylabelsize = 15, ticksize = 10)
                ax_hist.grid(True)

                cbar = plt.colorbar(h[3], ax=plt.gca(), pad=0.12)
                cbar.set_label('Density', labelpad=7, fontsize=12)
                cbar.ax.yaxis.set_label_position('left')
            
            ax_time = fig.add_subplot(gs[1, :])

            if auroral_index == 'AL_INDEX':
                model_data[model_list[0]][f'{auroral_index}_real'] = -1 * model_data[model_list[0]][f'{auroral_index}_real'] 

            ax_time.plot(model_data[model_list[0]]['Epoch'], model_data[model_list[0]][f'{auroral_index}_real'], color = 'black', label = 'Real', linewidth = 1.5)

            color_list = ['teal', 'red', 'green']
            for j, model in enumerate(model_list):
                ax_time.plot(model_data[model]['Epoch'], model_data[model][f'{auroral_index}_pred'], color = color_list[j], label = model, linewidth = 1.5)
            
            ax_time.set_xlim(start_time, end_time)
            
            ax_time.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
            plt.step(ax_time.xaxis.get_majorticklocs())

            ax_time.set_title(f'Comparison of Models for {auroral_index.replace("_INDEX", " Index")} (Delay {delay} min)', fontsize = 10)
            setup_axis_style(ax_time, xlabel = 'Date', ylabel = format_parameter_label(auroral_index), xlabelsize = 15, ylabelsize = 15, ticksize = 10)
            ax_time.legend(loc = 'best')
            ax_time.grid(True)

            plt.tight_layout()

            filname = f'Comparison_{auroral_index.replace("_INDEX", "Index")}_{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}_Delay_{delay}.png'

            plt.savefig(comparison_file + filname)
            plt.close()


#* Gift Plot
def gif_plot(result_file, processed_file, gifs_file, auroral_index, delay_length, frame_steps = 2):
    """
    Generates a GIF showing the temporal evolution of auroral index predictions for different models.

    Args: 
        result_file : str
            Path to the directory containing the model results.
        processed_file : str
            Path to the directory containing the storm list CSV file.
        gifs_file : str
            Path to the directory where GIFs will be saved.
        auroral_index : str
            The name of the auroral index being predicted.
        delay_length : list
            List of delay values used in the model.
        frame_steps : int, optional
            Number of steps to skip between frames in the GIF (Default is 2).
    """
    storm_list = load_storm_data(processed_file)
    model_list = ['LSTM', 'CNN', 'ANN']

    k = int(2 * np.sqrt(len(storm_list)))

    for i, storm in enumerate(storm_list['Epoch']):
        start_time = storm - pd.Timedelta('24h')
        end_time = storm + pd.Timedelta('24h')

        for delay in delay_length:
            model_data = {}

            for model in model_list:
                result_file = f'{result_file}_results_delay_{delay}_{auroral_index}_{model}.feather'
                metric_file = f'{result_file}_metrics_delay_{delay}_{auroral_index}_{model}.csv'

                df = pd.read_feather(result_file)
                df_metric = pd.read_csv(metric_file)

                storm_df = df[(df['Epoch'] >= start_time) & (df['Epoch'] <= end_time)]
                model_data[model] = storm_df
            
                frames = []

            all_timestamps = model_data['ANN']['Epoch'].values
            frame_timestamps = all_timestamps[::frame_steps]

            for i, current_time in enumerate(frame_timestamps):
                current_time_pd = pd.Timestamp(current_time)

                fig = plt.figure(figsize = (18,10))
                title = f'Prediction from {start_time.strftime("%Y-%m-%d")} to {end_time.strftime("%Y-%m-%d")} for the {auroral_index.replace("_INDEX", " Index")}'
                fig.suptitle(title, fontsize=15, fontweight='bold')
                fig.subplots_adjust(top=0.92)
                gs = GridSpec(2, 3, figure = fig, height_ratios = [1, 2])
                gs.update(wspace = 0.3, hspace = 0.3)

                for j, model in enumerate(model_list):
                    ax_hist = fig.add_subplot(gs[0, j])

                    current_data = model_data[model][model_data[model]['Epoch'] <= current_time_pd]

                    if len(current_data) > 5:
                        real = np.log(np.abs(model_data[model][f'{auroral_index}_real'].values))
                        pred = np.log(np.abs(model_data[model][f'{auroral_index}_pred'].values))

                        h = ax_hist.hist2d(real, pred, bins = k, cmap = 'viridis', norm = LogNorm())

                        p_min = min(min(real), min(pred))
                        p_max = max(max(real), max(pred))   
                        ax_hist.plot([p_min, p_max], [p_min, p_max], color = 'black', linewidth = 2)

                        r = df_metric['R_Score'].values[0]

                        ax_hist.set_title(f'{model} - R = {r:.2f}', fontsize = 15, fontweight = 'bold')
                        setup_axis_style(ax_hist, xlabel = 'Log(Real)', ylabel = 'Log(Pred)', xlabelsize = 15, ylabelsize = 15, ticksize = 10)
                        ax_hist.grid(True)

                        cbar = plt.colorbar(h[3], ax=plt.gca(), pad=0.12)
                        cbar.set_label('Density', labelpad=7, fontsize=12)
                        cbar.ax.yaxis.set_label_position('left')
                    else:
                        ax_hist.set_title(f'{model} - No Data', fontsize = 15, fontweight = 'bold')
                    
            ax_time = fig.add_subplot(gs[1, :])

            if auroral_index == 'AL_INDEX':
                model_data[model_list[0]][f'{auroral_index}_real'] = -1 * model_data[model_list[0]][f'{auroral_index}_real'] 

            for j, model in enumerate(model_list):
                current_data = model_data[model][model_data[model]['Epoch'] <= current_time_pd]
            
                if j == 0:
                    ax_time.plot(current_data['Epoch'], current_data[f'{auroral_index}_real'], color = 'black', label = 'Real', linewidth = 1.5)
                
                color = ['teal', 'red', 'green'][j]
                ax_time.plot(current_data['Epoch'], current_data[f'{auroral_index}_pred'], color = color, label = model, linewidth = 1.5)

            ax_time.set_xlim(start_time, end_time)

            ax_time.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
            plt.step(ax_time.xaxis.get_majorticklocs())

            time_str = current_time_pd.strftime('%H:%M')
            ax_time.text(0.5, 0.5, f'Time: {time_str}', transform=ax_time.transAxes, ha = 'center', fontsize = 12, bbox = dict(facecolor = 'white', alpha = 0.5))

            ax_time.set_title(f'Temporal Evolution - {auroral_index.replace("_INDEX", " Index")} (Delay {delay} min)', fontsize = 10)
            setup_axis_style(ax_time, xlabel = 'Date', ylabel = format_parameter_label(auroral_index), xlabelsize = 15, ylabelsize = 15, ticksize = 10)
            ax_time.legend(loc = 'best')
            ax_time.grid(True)

            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format = 'png', dpi = 150, bbox_inches = 'tight')
            buf.seek(0)

            frame = imageio.imread(buf)
            frames.append(frame)

            plt.close()
        
        filname = f'Temporal_evolution_{auroral_index.replace("_INDEX", "Index")}_{start_time.strftime("%Y%m%d")}_{end_time.strftime("%Y%m%d")}_Delay_{delay}.gif'
        duration = 0.5 if len(frames) > 50 else 0.7
        imageio.mimsave(gifs_file + filname, frames, duration = duration, loop = 0)




 








