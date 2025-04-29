from variables import *
from folders import *
from data_processing import *
from models import *
from model_training import *
from plot import *


def main():
    """
    Main function to run the auroral prediction project.
    """
    
    #* Create Project Structure
    structure_proyect(project_file, files_plots, files)

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    #* Load and Process Data
    df = dataset(in_year, out_year, omni_file, raw_file, processOMNI)
    df_storms = storm_selection(df, processed_file)

    #* Process OMNI plot data
    if processPLOT:
        list_dir = os.listdir(auroral_historic_file)
        if not list_dir:
            time_serie_plot(df_storms, omni_param, auroral_param, processed_file, auroral_historic_file, solar_historic_file)
            correlation_plot(df_storms, corr_file)
    

    #* Create Dataset
    df_scaler = scaler_df(df_storms, scaler_type, auroral_param, omni_param)
    train_df, val_df, test_df = create_set_prediction(df_scaler, set_split, test_size, val_size)

    del df, df_storms, df_scaler

    metric_test_delay = pd.DataFrame()

    #* Create Dataset for each delay length
    for delay in delay_length:
        train_solar, train_index = time_delay(train_df, omni_param, auroral_index, delay, type_model, 'train')
        val_solar, val_index = time_delay(val_df, omni_param, auroral_index, delay, type_model, 'val')
        test_solar, test_index, test_epoch = time_delay(test_df, omni_param, auroral_index, delay, type_model, 'test')

        #* Create Torch set 
        train_torch = DataTorch(train_solar, train_index, device)
        val_torch = DataTorch(val_solar, val_index, device)
        test_torch = DataTorch(test_solar, test_index, device)

        train_loader = DataLoader(train_torch, batch_size = batch_train, shuffle = True)
        val_loader = DataLoader(val_torch, batch_size = batch_val, shuffle = False)
        test_loader = DataLoader(test_torch, batch_size = batch_test, shuffle = False)

        #* Select Model
        model = type_nn(type_model, train_solar, drop, kernel_cnn, num_layer_lstm, delay, device)
        criterion = nn.MSELoss()

        #* Training Model
        model, metrics_df = train_model(model, criterion, optimizer_type, train_loader, val_loader, EPOCH, lr, delay, type_model, auroral_index, schler, patience_schler, device, model_file)
        metrics_plot(metrics_df, delay, auroral_index, type_model, train_loss_file, train_r_score_file, train_d2_abs_file, train_d2_tweedie_file)

        #* Testing Model and save results
        result_df, metric_test = model_testing(model, criterion, test_loader, model_file, type_model, auroral_index, delay, test_epoch, device)
        result_df.to_feather(f'{result_file}result_delay_{delay}_{auroral_index}_{type_model}.feather')
        metric_test.to_csv(f'{result_file}metrics_delay_{delay}_{auroral_index}_{type_model}.csv')

        metric_test = metric_test.rename(columns={col: f"{col}_{delay}" for col in metric_test.columns})
        metric_test_delay = pd.concat([metric_test_delay, metric_test], axis = 1, ignore_index = False)

    delay_metrics_plot(metric_test_delay, delay_length, auroral_index, type_model, test_loss_file, test_r_score_file, test_d2_abs_file, test_d2_tweedie_file)

       


if __name__ == '__main__':
    main()