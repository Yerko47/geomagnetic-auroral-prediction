from variables import *
from folders import *
from data_processing import *
from models import *
from model_training import *


def main():
    """
    
    """
    print('Start Training Model')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    structure_proyect(proyect_file, files)
    df = dataset(in_year, out_year, omni_file, raw_file, processOMNI)
    df = storm_selection(df, processed_file)
    df = scaler_df(df, scaler, auroral_param, omni_param)

    train_df, val_df, test_df = create_set_prediction(df, set_split, test_size, val_size)
    
    for delay in delay_length:
        print(f'\n[   Training for delay: {delay}   ]')
        train_omni, train_index = time_delay(train_df, omni_param, auroral_index, delay, type_model, "train")
        val_omni, val_index = time_delay(val_df, omni_param, auroral_index, delay, type_model, "val")
        test_omni, test_index, test_epoch = time_delay(test_df, omni_param, auroral_index, delay, type_model, "test")

        train_loader = DataLoader(DataTorch(train_omni, train_index, device), 
                               batch_size=batch_train, shuffle=True)
        val_loader = DataLoader(DataTorch(val_omni, val_index, device), 
                               batch_size=batch_val, shuffle=False)
        test_loader = DataLoader(DataTorch(test_omni, test_index, device), 
                               batch_size=batch_test, shuffle=False)

        model = type_nn(type_model, train_omni, drop, kernel_cnn, num_layer_lstm, delay, device)
        criterion = nn.MSELoss()
       
        set_seed(42 + delay) 
            
        model, metrics_df = train_model(
            model, criterion, optimizer, train_loader, val_loader, EPOCH,
            lr, delay, type_model, auroral_index, schler, patience_schler,
            device, model_file
        )
                
        result_df, test_metrics = model_testing(model, criterion, test_loader, model_file, 
                                               type_model, auroral_index, delay, test_epoch, device)
        result_df.to_feather(f"{test_file}results_delay_{delay}_{auroral_index}_{type_model}.feather")
        test_metrics.to_csv(f"{test_file}metrics_delay_{delay}_{auroral_index}_{type_model}.csv")        


if __name__ == '__main__':
    main()