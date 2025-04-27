"""
Project variables
"""

#* DATE
in_year = 1995
out_year = 2018


#* AURORAL ELECTROJET INDEX
auroral_index = "AL_INDEX"                  # AE_INDEX    |    AL_INDEX    |    AU_INDEX


#* SPLIT TRAIN/VAL/TEST SET
set_split = "random"                     # organized --> Temporary order    |    random --> Stratified random split

test_size = 0.2
val_size = 0.2


#* PARAMETERS
omni_param = ['B_Total',
              'BX_GSE',
              'BY_GSE',
              'BZ_GSE',
              'BY_GSM',
              'BZ_GSM',
              'flow_speed',
              'Vx',
              'Vy',
              'Vz',
              'T',
              'Pressure',
              'E_Field'
              ]

auroral_param = ['AE_INDEX',
                 'AU_INDEX',
                 'AL_INDEX',
                 ]


#* NEURAL NETWORK GENERAL PARAMETERS
type_model = "LSTM"                       # ANN    |    LSTM    |    CNN    |    etc...
scaler = "robust"                         # robust --> RobustScaler()    |   standard --> StandardScaler()    |   minmax --> MinMaxScaler()
delay_length = [30, 40, 50, 60]

# Loader
n = 2
batch_test = 520
batch_val = batch_test * n
batch_train = batch_test * n

# Hyperparametrs
EPOCH = 100
lr = 1e-3
patience = 20
drop = 0.2
optimizer = "Adam"                       # Adam --> optimizer.Adam()    |   SGD --> optimizer.SDG()
schler = "Cosine"                       # Reduce --> lr.scheduler.ReduceLROnPlateau()    |   Cosine --> lr_scheduler.CosineAnnealingLR()
patience_schler = 50

# NN particular parameters
num_layer_lstm = 1
kernel_cnn = 7 

project_file = f'/home/yerko/Desktop/Projects/auroral_prediction/'
omni_files = f'/data/omni/hro_1min/'
 
files = ['data/raw/', 'data/processed/',
        'models/', 'notebooks/',
        'src/', 'tests/', 'docs/', 
        'plots/']

files_plots = ['historic/index/', 'historic/solar/', 'stadistics/correlation/',
               'stadistics/others/', 'training/loss/', 'training/r_score/',
               'test_plot/metrics/loss/', 'test_plot/metrics/r_score/', 
               'test_plot/density/', 'test_plot/date/', 'comparison/', 
               'gifs/']

raw_file = project_file + files[0]
processed_file = project_file + files[1]
model_file = project_file + files[2]
test_file = project_file + files[5]
plots_file = project_file + project_file[6]

auroral_historic_file = project_file + files_plots[0]
solar_historic_file = project_file + files_plots[1]
corr_file = project_file + files_plots[2]
train_loss_file = project_file + files_plots[4]
train_r_score_file = project_file + files_plots[5]
test_loss_file = project_file + files_plots[6]
test_r_score_file = project_file + files_plots[7]
test_density_file = project_file + files_plots[8]
test_date_file = project_file + files_plots[9]
comparison_file = project_file + files_plots[10]
gifs_file = project_file + files_plots[11]






processOMNI = True