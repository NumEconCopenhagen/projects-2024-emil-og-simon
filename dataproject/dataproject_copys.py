import pandas as pd
from dstapi import DstApi
import matplotlib.pyplot as plt
import ipywidgets as widgets

# IMPORTING AND DISPLAYING THE DATASET (DEMO4)

def import_demo4_data():
    # Importing the dataset
    demo4_api = DstApi('DEMO4')

    # Displaying the data summary
    tabsum2 = demo4_api.tablesummary(language='en')
    display(tabsum2)

    # Overview of the available values for each variable
    for variable in tabsum2['variable name']:
        print(variable + ':')
        display(demo4_api.variable_levels(variable, language='en'))

    # Defining a dictionary for the parameters
    params_demo4 = demo4_api._define_base_params(language='en')

    # Selecting all parameters for now
    params_demo4 = {
        'table': 'demo4',
        'format': 'BULK',
        'lang': 'en',
        'variables': [
            {'code': 'REGION', 'values': ['*']},
            {'code': 'BRANCHEDB0710', 'values': ['*']},
            {'code': 'MÆNGDE4', 'values': ['*']},
            {'code': 'Tid', 'values': ['*']}
        ]
    }

    # Loading the data from dst
    demo4_all = demo4_api.get_data(params=params_demo4)
    return demo4_all

# CLEANING AND RENAMING THE DATASET (DEMO4) 

def clean_demo4_data(demo4_all):
    # Renaming the columns
    demo4_all.rename(columns={
        'REGION': 'regions',
        'BRANCHEDB0710': 'industry',
        'MÆNGDE4': 'x',
        'TID': 'year',
        'INDHOLD': 'total'
    }, inplace=True)

    # Selecting the parameters we want to examine
    I = demo4_all['regions'].str.contains('All Denmark') & demo4_all['industry'].str.contains('TOT Industry total')
    demo4_reduce = demo4_all.loc[I, :].copy()  # Make a copy to avoid SettingWithCopyWarning

    # Resetting the index
    demo4_reduce.reset_index(inplace=True, drop=True)

    # Deleting the column "regions" since it has been reduced to only containing "All Denmark"
    drop_this = ['regions']
    demo4_reduce.drop(drop_this, axis=1, inplace=True)

    print("Cleaning has been done successfully.")
    return demo4_reduce

if __name__ == "__main__":
    demo4_all_data = import_demo4_data()
    cleaned_demo4_data = clean_demo4_data(demo4_all_data)
    display(cleaned_demo4_data)

# MERGING THE DATASETS (DEMO4 AND GF02)

def merge_datasets(demo4_all, gf02_industries_reduced):
    # Creating a subset of DEMO4 including the industries
    I = demo4_all['regions'].str.contains('All Denmark')
    demo4_industries = demo4_all.loc[I, :].copy()  # Make a copy to avoid SettingWithCopyWarning

    # Resetting the index
    demo4_industries.reset_index(inplace=True, drop=True)

    # Dropping the 'regions' column
    drop_this = ['regions']
    demo4_industries.drop(drop_this, axis=1, inplace=True)
    
    # Pivoting the dataframe to unpack the values of 'x' into individual columns
    demo4_unpacked_merge = demo4_industries.pivot(index=['industry', 'year'], columns='x', values='total').reset_index()
    demo4_unpacked_merge.columns.name = None

    # Sorting and resetting the index of gf02_industries
    gf02_industries_merge = gf02_industries_reduced.sort_values(['industry', 'year']).reset_index(drop=True)

    # Performing an outer merge on industry and year
    merged_data = pd.merge(gf02_industries_merge, demo4_unpacked_merge, on=['industry', 'year'], how='outer')

    # Renaming the columns
    merged_data.rename(columns={
        'Employees (in full-time persons)': 'employees(fulltime)',
        'Export (DKK 1000)': 'Export(DKK1000)',
        'New enterprises (number)': 'Startups',
        'Terminated enterprises (number)': 'Bankruptcies',
        'Turnover (DKK 1000)': 'Turnover(DKK1000)'
    }, inplace=True)

    return merged_data

if __name__ == "__main__":
    demo4_all_data = import_demo4_data()
    cleaned_demo4_data = clean_demo4_data(demo4_all_data)
    display(cleaned_demo4_data)

    # Assuming gf02_industries_reduced is already available as a DataFrame
    # If not, you need to load and prepare it similarly
    # gf02_industries_reduced = ...

    # Merging datasets
    gf02_demo4_merged = merge_datasets(demo4_all_data, gf02_industries_reduced)
    display(gf02_demo4_merged)

# ANALYSIS

# PLOTTING FUNCTION WITH WIDGETS
def merged_plot(data):
    #We define a function, where we filter by size and industry, so we can construct a widget with two dropdowns below.
    def plot_gf02_all_rename(df, size, industry): 
        filtered_df = df[(df['size'] == size) & (df['industry'] == industry)]
        ax=filtered_df.plot(x='year', y='companies', style='-o', legend=False)

    widgets.interact(plot_gf02_all_rename, 
    df = widgets.fixed(data),
    industry=widgets.Dropdown(description='Industry', options=data['industry'].unique(), value='1 Agriculture, forestry and fishing'),
    size = widgets.Dropdown(description='Size', options=data['size'].unique(), value='Total, all enterprises')
);