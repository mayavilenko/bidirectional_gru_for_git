from model.config import *

son_parent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/parent_child_dict.pickle'
category_weight_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/category_weight_dict.pickle'
son_parent_corr_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/parent_child_corr_dict.pickle'

train_dataset_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/reversed_hgru_with_corr_1_period/data/train_dataset.pickle'
test_dataset_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/reversed_hgru_with_corr_1_period/data/test_dataset.pickle'
coefficient_dict_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/reversed_hgru_with_corr_1_period/data/coefficient_dict.pickle'

category_id_to_category_name_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/category_id_to_category_name_dict.pickle'
categories_per_indent_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/categories_per_indent_dict.pickle'

weightspath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/reversed_hgru_with_corr_1_period/models_weights/'

test_predictions_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/reversed_hgru_with_corr_1_period/test_predictions.pickle'
test_predictions_path_with_hgru = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/mayas_project/reversed_hgru_with_corr_1_period/test_predictions_with_hgru.pickle'


reversed_parent_son_corr_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/reversed_parent_son_corr_dict.pickle'
reversed_parent_son_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/reversed_parent_son_dict.pickle'
reversed_parent_to_son_list_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/reversed_parent_to_son_list_dict.pickle'
sgru_weight_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/sgru_model_weights.pickle'
hgru_weight_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hgru_model_weights.pickle'


HRNNpath = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_new.pickle'
bi_directional_path = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_dataset_dict.pickle'

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