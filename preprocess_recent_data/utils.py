#imports:
import numpy as np
import pandas as pd
import math
import datetime
from datetime import datetime

from config import *

########################################################################################################################################################################

def add_parent_and_ids(df):
    old_data = pd.read_csv(OLD_PATH)
    old_df = old_data[old_data['Date']=='2019-03-15'][['Category','Category_id','Parent','Parent_ID','Weight']].drop_duplicates()
    old_df['Category'] = old_df['Category'].apply(lambda x: x.replace("'","’"))
    old_df.set_index(['Category'], inplace = True)
    old_dict = old_df.to_dict('index')
    
    for new_category in new_to_old_category_dict.keys():
        df.loc[df['Category'] == new_category, 'Category'] = new_to_old_category_dict[new_category]
    
    for key in old_dict.keys():
        df.loc[df['Category']==key, 'Category_id'] = old_dict[key]['Category_id']
        df.loc[df['Category']==key, 'Parent'] = old_dict[key]['Parent']
        df.loc[df['Category']==key, 'Parent_ID'] = old_dict[key]['Parent_ID']
        df.loc[df['Category']==key, 'Weight'] = old_dict[key]['Weight']

    
    # edge case - gasoline:
    df.loc[df['Category']=='Gasoline', 'Category'] = 'Gasoline all types'
    df.loc[df['Category']=='Gasoline', 'Category_id'] = 789.0
    df.loc[df['Category']=='Gasoline', 'Parent'] = 'Motor fuel'
    df.loc[df['Category']=='Gasoline', 'Parent_ID'] = 3840.0
    
    df.loc[df['Category']=='Gasoline all types', 'Category_id'] = 789.0
    df.loc[df['Category']=='Gasoline all types', 'Parent'] = 'Motor fuel'
    df.loc[df['Category']=='Gasoline all types', 'Parent_ID'] = 3840.0
    
    return df

def create_post_2019_df(years = ['2019','2020','2021','2022','2023'], final_lst = []):
    for year in years:
        if year == '2023':
            month = '09'
        else:
            month = '12'
        raw_data = pd.read_excel(NEW_PATH+year+month+'.xlsx', header = header, skipfooter = skipfooter)
        raw_data.reset_index(inplace = True)
        raw_data = raw_data.iloc[:,columns_needed_dict[year]]
        raw_data.columns = pre_melt_column_dict[year]
        raw_data['Category'] = raw_data['Category'].apply(lambda x: str(x))
        raw_data['Category'] = raw_data['Category'].apply(lambda x: x.replace("'","’"))
        raw_data['Category'] = raw_data['Category'].apply(lambda x: x.strip('()0123456789./'))
        raw_data = pd.melt(raw_data, id_vars = ['Indent','Category','Weight']).rename(columns={'variable':'Date','value':'Price'})
        final = add_parent_and_ids(raw_data)
        final = final[(final['Category']=='All items')|(~(final['Parent'].isna()))]
        final_lst.append(final)

    post_2019_df = pd.concat(final_lst)
    
    return post_2019_df