#from model.config import *

son_parent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/parent_child_dict.pickle'
category_weight_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/category_weight_dict.pickle'
son_parent_corr_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/parent_child_corr_dict.pickle'
train_dataset_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/hgru_model/data/train_dataset.pickle'
test_dataset_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/hgru_model/data/test_dataset.pickle'
category_id_to_category_name_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/category_id_to_category_name_dict.pickle'
categories_per_indent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/categories_per_indent_dict.pickle'
weightspath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/hgru_model/models_weights/'
test_predictions_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/hgru_model/test_predictions.pickle'

HRNNpath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_new.pickle'

HRNN_EXPANDING_HOR_1 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_1.pickle'
HRNN_EXPANDING_HOR_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_2.pickle'
HRNN_EXPANDING_HOR_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_3.pickle'
HRNN_EXPANDING_HOR_4 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_4.pickle'
HRNN_EXPANDING_HOR_5 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_5.pickle'
HRNN_EXPANDING_HOR_6 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_6.pickle'

plot_dict = {'Inflation t+1' : 'Actual - Period 1', 
            'Inflation t+2' : 'Actual - Period 2',
            'Inflation t+3' : 'Actual - Period 3',
            'Inflation t+4' : 'Actual - Period 4',
            'Inflation t+5' : 'Actual - Period 5',
            'Inflation t+6' : 'Actual - Period 6',
            'Inflation t+7' : 'Actual - Period 7',
            'expanding horizon: 0' : 'Prediction - Period 1',
            'expanding horizon: 1' : 'Prediction - Period 2',
            'expanding horizon: 2' : 'Prediction - Period 3',
            'expanding horizon: 3' : 'Prediction - Period 4',
            'expanding horizon: 4' : 'Prediction - Period 5',
            'expanding horizon: 5' : 'Prediction - Period 6',
            'expanding horizon: 6' : 'Prediction - Period 7'}