
columns_order = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Indent', 'Weight', 'Parent', 'Parent_ID']
columns_order_horizon = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Indent', 'Weight', 'Parent', 'Parent_ID']

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# US
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
US_HRNN_PATH = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/resources/hrnn_cpi_dataset.csv'
US_SYN_PATH = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/resources/synthetic_us_cpi_dataset.csv'
US_bi_directional_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/bi_directional_us_dataset_dict.pickle'
US_syn_bi_directional_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/bi_directional_syn_us_dataset_dict.pickle'
#US_bi_directional_5_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/bi_directional_us_5_dataset_dict.pickle'
#US_bi_directional_4_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/bi_directional_us_4_dataset_dict.pickle'

us_parent_son_pickle =  '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_parent_dict.pickle'
us_parent_son_name_pickle =  '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_parent_name_dict.pickle'
us_cat_weight_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_category_weight_dict.pickle'
us_category_id_to_name_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_category_id_to_category_name_dict.pickle'
us_category_id_per_indent_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_categories_per_indent_dict.pickle'
us_parent_to_sons_list_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/us_parent_to_son_list_dict.pickle'

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Canada
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
CANADA_HRNN_PATH = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/resources/canada_cpi_dataset_linear_regression_weights.csv'
canada_bi_directional_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/bi_directional_canada_dataset_dict.pickle'

canada_parent_son_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_parent_dict.pickle'
canada_parent_son_name_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_parent_name_dict.pickle'
canada_cat_weight_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_category_weight_dict.pickle'
canada_category_id_to_name_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_category_id_to_category_name_dict.pickle'
canada_category_id_per_indent_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_categories_per_indent_dict.pickle'
canada_parent_to_sons_list_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/canada_parent_to_sons_list_dict.pickle'
 
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Norway
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
NORWAY_HRNN_PATH = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/resources/norway_cpi_dataset.csv'
norway_bi_directional_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/bi_directional_norway_dataset_dict.pickle'

norway_parent_son_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_parent_dict.pickle'
norway_parent_son_name_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_parent_name_dict.pickle'
norway_cat_weight_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_category_weight_dict.pickle'
norway_category_id_to_name_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_category_id_to_category_name_dict.pickle'
norway_category_id_per_indent_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_categories_per_indent_dict.pickle'
norway_parent_to_sons_list_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/hgru_clean/pickle_files/norway_parent_to_sons_list_dict.pickle'