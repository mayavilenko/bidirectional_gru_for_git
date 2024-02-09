import torch.nn as nn
import torch
import numpy as np
import random

torch.manual_seed(1)
np.random.seed(2)
random.seed(3)

SequenceLength = 13
Features = 1
OutputDim = 1
HiddenSize = 64
LayersDim = 1
DropoutProb = 0.0
#Lr = 0.001
Lr = 0.09782661222201432
Criterion = nn.MSELoss()
Epochs = 500
BatchSize = 32
Year = 2020

TbDirectory = 'tbs/'
CheckpointPath = 'checkpoints/all_checkpoints/'
BestcheckpointPath = 'checkpoints/best_checkpoints/'

BaselinePath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_canada_dataset_dict.pickle'
train_dataset_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/basic_model_canada/data/train_dataset.pickle'
test_dataset_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/basic_model_canada/data/test_dataset.pickle'
category_id_to_category_name_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/canada_category_id_to_category_name_dict.pickle'
test_predictions_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/basic_model_canada/test_predictions.pickle'

