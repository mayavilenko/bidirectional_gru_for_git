from model.GRU_model import *
import torch.nn as nn
import torch

SequenceLength = 13
Features = 1
OutputDim = 1
HiddenSize = 32
LayersDim = 1
DropoutProb = 0.0
Lr = 0.05
Criterion = nn.MSELoss()
Epochs = 250
BatchSize = 32
Year = 2020

TbDirectory = 'tbs/'

train_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/data/train_dataset.pickle'
test_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/data/test_dataset.pickle'
category_id_to_category_name_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_category_id_to_category_name_dict.pickle'
categories_per_indent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_categories_per_indent_dict.pickle'
son_parent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_parent_dict.pickle'
weightspath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/hgru_ensemble/models_weights'
test_predictions_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/US/hgru_ensemble/test_predictions.pickle'

#Define our device
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

Model=GRUModel(input_dim=Features, hidden_dim=HiddenSize, layer_dim=LayersDim, output_dim=OutputDim, dropout_prob=DropoutProb, seed=42)
Optimizer=torch.optim.AdamW(Model.parameters(), lr=Lr)

