import sys
sys.path.insert(0, '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/')
from model.GRU_model import *
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

SequenceLength = 13
Features = 1
OutputDim = 1
HiddenSize = 64
LayersDim = 1
DropoutProb = 0.0
Lr = 0.1
Criterion = nn.MSELoss()
Epochs = 100
BatchSize = 32
Year = 2020
loss_coef_1= 1*np.exp(-10)
loss_coef_2= 1*np.exp(-10)
loss_coef_3= 6.16933911191657*np.exp(-8)
alpha= 1.0

TbDirectory = 'tbs/'

train_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/data/train_dataset.pickle'
test_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/data/test_dataset.pickle'
category_id_to_category_name_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_category_id_to_category_name_dict.pickle'
categories_per_indent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_categories_per_indent_dict.pickle'
son_parent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_parent_dict.pickle'
weightspath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/models_weights/'
parent_to_son_list_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_parent_to_sons_list_dict.pickle'
sgru_model_weights_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_sgru_model_weights.pickle'
hgru_model_weights_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_hgru_model_weights.pickle'
coefficient_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/data/coefficient_dict.pickle'

# Loss Analysis params:
weightspath_1 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/models_weights_1/'
weightspath_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/models_weights_2/'
weightspath_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/models_weights_3/'
weightspath_1_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/models_weights_1_2/'

test_predictions_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/test_predictions.pickle'
test_predictions_path_1 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/test_predictions_1.pickle'
test_predictions_path_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/test_predictions_2.pickle'
test_predictions_path_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/test_predictions_3.pickle'
test_predictions_path_1_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Canada/bidirectional/test_predictions_1_2.pickle'


#Define our device
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

Model=GRUWithAttentionModel(input_dim=Features, hidden_dim=HiddenSize, layer_dim=LayersDim, output_dim=OutputDim, dropout_prob=DropoutProb)
Optimizer=torch.optim.AdamW(Model.parameters(), lr=Lr)
