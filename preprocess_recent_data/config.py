NEW_PATH = './new_raw_data/cpi-u-'
OLD_PATH = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/resources/cpi_us_dataset.csv'
POST_2019_PATH = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/resources/post_2019_df.csv'
POST_2019_PATH_new = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/resources/post_2019_df_new.csv'


header = 6
skipfooter = 20 

columns_needed_dict = {  '2019': [1,2,3,18,19,20,21,22,23,24,25,26,27,28,29]
                        ,'2020': [1,2,3,18,19,20,21,22,23,24,25,26,27,28,29]
                        ,'2021': [1,2,3,18,19,20,21,22,23,24,25,26,27,28,29]
                        ,'2022': [1,2,3,18,19,20,21,22,23,24,25,26,27,28,29]
                        ,'2023': [1,2,3,21,22,23,24,25,26,27,28,29]
                        }

pre_melt_column_dict = {'2019': ['Indent', 'Category', 'Weight','2019-01-15', '2019-02-15', '2019-03-15','2019-04-15','2019-05-15','2019-06-15','2019-07-15','2019-08-15','2019-09-15','2019-10-15', '2019-11-15','2019-12-15']
                      , '2020': ['Indent', 'Category', 'Weight','2020-01-15', '2020-02-15', '2020-03-15','2020-04-15','2020-05-15','2020-06-15','2020-07-15','2020-08-15','2020-09-15','2020-10-15', '2020-11-15','2020-12-15']
                      , '2021': ['Indent', 'Category', 'Weight','2021-01-15', '2021-02-15', '2021-03-15','2021-04-15','2021-05-15','2021-06-15','2021-07-15','2021-08-15','2021-09-15','2021-10-15', '2021-11-15','2021-12-15']
                      , '2022': ['Indent', 'Category', 'Weight','2022-01-15', '2022-02-15', '2022-03-15','2022-04-15','2022-05-15','2022-06-15','2022-07-15','2022-08-15','2022-09-15','2022-10-15', '2022-11-15','2022-12-15']
                      , '2023': ['Indent', 'Category', 'Weight','2023-01-15', '2023-02-15', '2023-03-15','2023-04-15','2023-05-15','2023-06-15','2023-07-15','2023-08-15','2023-09-15']
                      }

new_to_old_category_dict = {
    #'Meats, poultry, and fish':'Meats, poultry, fish, and eggs',
    'Other pork including roasts, steaks, and ribs':'Other pork including roasts and picnics',
    'Instant coffee':'Instant and freeze dried coffee',
    'Sugar and sugar substitutes':'Sugar and artificial sweeteners',
    'Gasoline (all types' : 'Gasoline all types',
    #'Utility (piped) gas service' : 'Fuels and utilities',
    "Women’s underwear, nightwear, swimwear, and accessories":"Women’s underwear, nightwear, sportswear and accessories",
    'Computers, peripherals, and smart home assistants':'Personal computers and peripheral equipment',
    #'Owners’ equivalent rent of residences':'Owners’ equivalent rent of primary residence',
    'Airline fares':'Airline fare',
    'Cable and satellite television service':'Cable and satellite television and radio service',
    'Video discs and other media, including rental of video':'Video discs and other media, including rental of video and audio',
    'Rental of video discs and other media':'Rental of video or audio discs and other media',
    'Photographers and photo processing':'Photographers and film processing',
    'Day care and preschool':'Child care and nursery school',
    'Residential telephone services':'Land-line telephone services'
    } 
#---------------------------------------------------------------------------------------------------------------------#
# STARTED CLEANUP HERE #
#---------------------------------------------------------------------------------------------------------------------#
# USA Baseline:
BASELINE_PATH = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/resources/baseline_cpi_dataset.csv'
columns_order = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Indent', 'Weight', 'Parent', 'Parent_ID']
columns_order_2 = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Inflation t+2', 'Indent', 'Weight', 'Parent', 'Parent_ID']
columns_order_3 = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Inflation t+2', 'Inflation t+3', 'Indent', 'Weight', 'Parent', 'Parent_ID']
columns_order_4 = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Inflation t+2', 'Inflation t+3', 'Inflation t+4', 'Indent', 'Weight', 'Parent', 'Parent_ID']
columns_order_8 = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Inflation t+2', 'Inflation t+3', 'Inflation t+4', 'Inflation t+5', 'Inflation t+6', 'Inflation t+7', 'Inflation t+8', 'Indent', 'Weight', 'Parent', 'Parent_ID']

BASELINE_PICKLE = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/baseline_dataset_dict.pickle'
BASELINE_PICKLE_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/baseline_dataset_dict_2_horizons.pickle'
BASELINE_PICKLE_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/baseline_dataset_dict_3_horizons.pickle'
BASELINE_PICKLE_4 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/baseline_dataset_dict_4_horizons.pickle'
BASELINE_PICKLE_8 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/baseline_dataset_dict_8_horizons.pickle'

# USA HRNN: 
# note: hrnn and baseline had different datasets since we removed some time periods for the hrnn and for the baseline we didnt need to since theres not dependency
HRNN_PATH = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/resources/hrnn_cpi_dataset.csv'
columns_order = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Indent', 'Weight', 'Parent', 'Parent_ID']
HRNN_PICKLE = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_new.pickle'

# USA BI-DIRECTIONAL: 
bi_directional_1_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_1_period_dataset_dict.pickle'
bi_directional_2_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_2_period_dataset_dict.pickle'
bi_directional_3_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_3_period_dataset_dict.pickle'
bi_directional_4_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_4_period_dataset_dict.pickle'
bi_directional_8_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_8_period_dataset_dict.pickle'

usa_parent_son_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/parent_child_dict.pickle'
usa_cat_weight_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/category_weight_dict.pickle'
usa_cat_id_to_name_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/category_id_to_category_name_dict.pickle'
usa_cats_per_indent_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/categories_per_indent_dict.pickle'
usa_parent_to_son_list = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/reversed_parent_to_son_list_dict.pickle'

# CANADA BI-DIRECTIONAL:
BI_DIRECTIONAL_CANADA_PICKLE = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_canada_dataset_dict.pickle'
parent_son_canada_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/canada_parent_son_dict.pickle'
category_id_to_name_canada_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/canada_category_id_to_category_name_dict.pickle'
category_id_per_indent_canada_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/canada_categories_per_indent_dict.pickle'
parent_to_sons_list_canada_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/canada_parent_to_sons_list_dict.pickle'
 
bi_directional_canada_2_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_canada_2_period_dataset_dict.pickle'
bi_directional_canada_3_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_canada_3_period_dataset_dict.pickle'
bi_directional_canada_4_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_canada_4_period_dataset_dict.pickle'
bi_directional_canada_8_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_canada_8_period_dataset_dict.pickle'

# NORWAY BI-DIRECTIONAL:
BI_DIRECTIONAL_NORWAY_PICKLE = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_norway_dataset_dict.pickle'
parent_son_norway_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/norway_parent_son_dict.pickle'
category_id_to_name_norway_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/norway_category_id_to_category_name_dict.pickle'
category_id_per_indent_norway_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/norway_categories_per_indent_dict.pickle'
parent_to_sons_list_norway_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/norway_parent_to_sons_list_dict.pickle'

bi_directional_norway_2_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_norway_2_period_dataset_dict.pickle'
bi_directional_norway_3_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_norway_3_period_dataset_dict.pickle'
bi_directional_norway_4_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_norway_4_period_dataset_dict.pickle'
bi_directional_norway_8_period_pickle = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/bi_directional_norway_8_period_dataset_dict.pickle'

#---------------------------------------------------------------------------------------------------------------------#
# STOPPED CLEANUP HERE #
#---------------------------------------------------------------------------------------------------------------------#

#PARENT_CHILD_CORR_PICKLE = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/parent_child_corr_dict.pickle'
#REVERSED_PARENT_SON_PICKLE = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/reversed_parent_son_dict.pickle'
#REVERSED_CORR_PICKLE = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/reversed_parent_son_corr_dict.pickle'

## Expanding Horizon
#HRNN_EXPANDING_HOR_1 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_1.pickle'
#HRNN_EXPANDING_HOR_2 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_2.pickle'
#HRNN_EXPANDING_HOR_3 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_3.pickle'
#HRNN_EXPANDING_HOR_4 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_4.pickle'
#HRNN_EXPANDING_HOR_5 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_5.pickle'
#HRNN_EXPANDING_HOR_6 = '/Users/mvilenko/Library/CloudStorage/OneDrive-PayPal/CPI_HRNN - version 2.0/pickle files/hrnn_dataset_dict_expanding_hor_6.pickle'


#bi_directional_columns_order = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Inflation t+2','Indent', 'Weight', 'Parent', 'Parent_ID']

#columns_order_hor1 = ['Category_id', 'Category', 'Year', 'Date', 'Price', 'Inflation t-12', 'Inflation t-11', 'Inflation t-10', 'Inflation t-9', 'Inflation t-8', 'Inflation t-7', 'Inflation t-6', 'Inflation t-5', 'Inflation t-4', 'Inflation t-3', 'Inflation t-2', 'Inflation t-1', 'Inflation t', 'Inflation t+1', 'Inflation t+2', 'Indent', 'Weight', 'Parent', 'Parent_ID']


