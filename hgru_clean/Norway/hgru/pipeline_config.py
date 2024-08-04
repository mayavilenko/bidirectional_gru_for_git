
import sys
sys.path.insert(0, '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/')
from model.GRU_model import *
import torch.nn as nn
import torch
import numpy as np

SequenceLength = 13
Features = 1
OutputDim = 1
HiddenSize = 64
LayersDim = 1
DropoutProb = 0.0
Lr = 0.00019837551214953915
Criterion = nn.MSELoss()
Epochs = 100
BatchSize = 32
Year = 2020

TbDirectory = 'tbs/'

train_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/data/train_dataset.pickle'
test_dataset_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/data/test_dataset.pickle'
category_id_to_category_name_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_category_id_to_category_name_dict.pickle'
categories_per_indent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_categories_per_indent_dict.pickle'
son_parent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_parent_dict.pickle'
weightspath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/hgru/models_weights/'
parent_to_son_list_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_parent_to_son_list_dict.pickle'
hgru_model_weights_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_hgru_model_weights.pickle'
sgru_model_weights_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_sgru_model_weights.pickle'
coefficient_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/data/coefficient_dict.pickle'

test_predictions_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/Norway/hgru/test_predictions.pickle'


#Define our device
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


Model=GRUModel(input_dim=Features, hidden_dim=HiddenSize, layer_dim=LayersDim, output_dim=OutputDim, dropout_prob=DropoutProb, seed=0)
Optimizer=torch.optim.AdamW(Model.parameters(), lr=Lr)
