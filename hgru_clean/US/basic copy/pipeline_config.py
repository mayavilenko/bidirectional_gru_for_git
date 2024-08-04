
import torch.nn as nn
import torch

SequenceLength = 13
Features = 1
OutputDim = 1
HiddenSize = 64
LayersDim = 1
DropoutProb = 0.0
Lr = 0.0758230781941514 #post-optuna
Criterion = nn.MSELoss()
Epochs = 500
BatchSize = 32
Year = 2020

TbDirectory = 'tbs/'
CheckpointPath = 'checkpoints/all_checkpoints/'
BestcheckpointPath = 'checkpoints/best_checkpoints/'

train_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/data/train_dataset.pickle'
test_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/data/test_dataset.pickle'
category_id_to_category_name_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_category_id_to_category_name_dict.pickle'
categories_per_indent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_categories_per_indent_dict.pickle'

#Define our device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

sgru_model_weights_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_sgru_model_weights.pickle'