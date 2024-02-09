import torch.nn as nn
import torch
import numpy as np
import random

torch.manual_seed(1)
np.random.seed(2)
random.seed(3)

SequenceLength = 13 #len of seq-not including prediction
Features = 1 #input size
OutputDim = 1 #output size
HiddenSize = 64 #The number of features in the hidden state h
LayersDim = 1
DropoutProb = 0.0
Lr = 0.001
Criterion = nn.MSELoss()
Epochs = 500
BatchSize = 32
Year = 2020

TbDirectory = 'tbs/'
CheckpointPath = 'checkpoints/all_checkpoints/'
BestcheckpointPath = 'checkpoints/best_checkpoints/'

BaselinePath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/baseline_dataset_dict.pickle'
HRNN_PATH = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_new.pickle'

son_parent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/parent_child_dict.pickle'
category_weight_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/category_weight_dict.pickle'
son_parent_corr_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/parent_child_corr_dict.pickle'
train_dataset_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/hgru_model/data/train_dataset.pickle'
test_dataset_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/hgru_model/datatest_dataset.pickle'
category_id_to_category_name_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/category_id_to_category_name_dict.pickle'
categories_per_indent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/categories_per_indent_dict.pickle'
weightspath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/hgru_model/models_weights/'
test_predictions_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/hgru_model/test_predictions.pickle'


exclude_categories = [
    'Film processing',
    'Entertainment',
    'Entertainment commodities',
    'Rent residential',
    "Men's footwear",
    'Housefurnishings',
    "Women's dresses",
    'Club dues and fees for participant sports and group exercises',
    'Housekeeping services', 'Gas piped and electricity',
    "Men's furnishings",
    'Audio discs, tapes and other media',
    'Maintenance and repairs',
    "Infants' equipment",
    'Infants’ equipment',
    'Other apparel commodities',
    'Intercity bus fare',
    "Men's suits, sport coats, and outerwear",
    'Household insurance',
    'Apparel commodities',
    'School books and supplies',
    "Physicians' services",
    'Olives, pickles, relishes',
    'All items less homeowners costs',
    'Homeowners costs',
    'Fuel oil and other household fuelcommodities',
    "Boys' apparel",
    "Women's footwear",
    'Entertainment services',
    'Other private transportationservices',
    'Professional medical services',
    'Nonalcoholic beverages',
    'Other renters costs',
    'Film and photographic supplies',
    'Renters costs',
    'Personal and educational expenses',
    'Personal and educational services',
    'Other private transportation',
    'Intercity train fare',
    'Toilet goods and personal careappliances',
    'Fuels', 'Automobile service clubs',
    "Men's pants and shorts",
    'Maintenance and repair services',
    "Men's shirts and sweaters",
    'Used cars',
    'Other private transportationcommodities',
    "Girls' apparel",
    "Women's outerwear",
    "Men's apparel",
    'Apparel services',
    "Women's apparel",
    'Other prepared food',
    'New cars and trucks',
    "Infants' furniture",
    'Gasoline',
    "Women's suits and separates",
    "Boys' and girls' footwear",
    "Women's underwear, nightwear, sportswear and accessories",
    'Information and information processingother than telephone services',
    'Utility',
    'Other poultry including turkey',
    'Lamb and organ meats',
    'Lamb and mutton',
]


