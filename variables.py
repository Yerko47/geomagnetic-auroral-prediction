"""
Project variables
"""

#* DATE
in_year = 1995
out_year = 2018


#* AURORAL ELECTROJET INDEX
auroral_index = "AE_INDEX"                  # AE_INDEX    |    AL_INDEX    |    AU_INDEX


#* SPLIT TRAIN/VAL/TEST SET
set_split = "organized"                     # organized --> Temporary order    |    random --> Stratified random split

test_size = 0.15
val_size = 0.3


#* PARAMETERS
omni_param = ['B_Total',
              'BZ_GSE',
              'BZ_GSM',
              'flow_speed',
              'Vx',
              'T',
              'Pressure',
              'E_Field',
              'IMF',
              'SYM_H'
              ]

auroral_param = ['AE_INDEX',
                 'AU_INDEX',
                 'AL_INDEX',
                 ]


#* NEURAL NETWORK GENERAL PARAMETERS
type_model = "CNN"                       # ANN    |    LSTM    |    CNN    |    etc...
scaler_type = "robust"                         # robust --> RobustScaler()    |   standard --> StandardScaler()    |   minmax --> MinMaxScaler()
delay_length = [30, 40, 50, 60]

# Loader
n = 2
batch_test = 1040
batch_val = batch_test * n
batch_train = batch_test * n

# Hyperparametrs
EPOCH = 200
lr = 1e-4
patience = 20
drop = 0.1
optimizer_type = "SGD"                       # Adam --> optimizer.Adam()    |   SGD --> optimizer.SDG()
schler = 'Cosine'                     # Reduce --> lr.scheduler.ReduceLROnPlateau    |   Cosine --> lr_scheduler.CosineAnnealingLR    |   CosineRW -- > lr_scheduler.CosineAnnealingWarmRestarts
patience_schler = 10

# NN particular parameters
num_layer_lstm = 1
kernel_cnn = 7

project_file = f'/home/yerko/Desktop/Projects/auroral_prediction/'
omni_file = f'/data/omni/hro_1min/'
 
files = ['data/raw/', 'data/processed/',
        'models/', 'models/results_csv/', 
        'src/', 'tests/', 'docs/', 'plots/'
        ]

files_plots = ['historic/index/', 'historic/solar/', 'stadistics/correlation/',
               'training/loss/', 'training/r_score/',
               'training/d2_abs/', 'training/d2_tweedie/',
               'test_plot/metrics/loss/', 'test_plot/metrics/r_score/',
               'test_plot/metrics/d2_abs/', 'test_plot/metrics/d2_tweedie/', 
               'comparison/', 'gifs/'
               ]

# Path general files
raw_file = project_file + files[0]
processed_file = project_file + files[1]
model_file = project_file + files[2]
result_file = project_file + files[3]
test_file = project_file + files[4]
plot_file = project_file + files[7]

# Path historic data
auroral_historic_file = plot_file + files_plots[0]
solar_historic_file = plot_file + files_plots[1]

# Path stadistics
corr_file = plot_file + files_plots[2]

# Path metrics training and validation process
train_loss_file = plot_file + files_plots[3]
train_r_score_file = plot_file + files_plots[4]
train_d2_abs_file = plot_file + files_plots[5]
train_d2_tweedie_file = plot_file + files_plots[6]

# Path metrics for diferent delay length in test process
test_loss_file = plot_file + files_plots[7]
test_r_score_file = plot_file + files_plots[8]
test_d2_abs_file = plot_file + files_plots[9]
test_d2_tweedie_file = plot_file + files_plots[10]

# Path test historic and density plots
comparison_file = plot_file + files_plots[11]

# Path gifs
gifs_file = plot_file + files_plots[12]






processOMNI = True
processPLOT = False