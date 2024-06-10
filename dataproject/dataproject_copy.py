import pandas as pd
from dstapi import DstApi
import matplotlib.pyplot as plt
import ipywidgets as widgets

# IMPORT, RENAME AND DISPLAY (GF02)

def import_overview_gf02(x):
    gf02_api = DstApi(x) #We start by extracting the data using the DstApi
    tabsum = gf02_api.tablesummary(language='en') #We then get an overview of what it contains by using the tablesummary-method
    for variable in tabsum['variable name']: #We want to use tabsum get a more detailed view of the variables.
        print(variable + ':')
        display(gf02_api.variable_levels(variable, language='en'))

    params_gf02 = gf02_api._define_base_params(language='en') #We do this to get an overview of the parameters we define bellow
    return gf02_api, params_gf02

def params_data_gf02(gf02_api): #We define the data we want using specific variable id's from above or extracting all using '*'.
    params_gf02 = {
        'table': 'gf02',
        'format': 'BULK',
        'lang': 'en',
        'variables': [ 
            {'code': 'KOMK', 'values': ['000']}, #id 000 --> we look at companies in all of Denmark
            {'code': 'BRANCHEDB0710TIL127', 'values': ['TOT', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']}, #we look at all industries (10-grp) in Denmark (1-11) and the total (TOT).
            {'code': 'FIRMSTR', 'values': ['*']},
            {'code': 'Tid', 'values': ['*']}
        ]
    }
    
    print("Parameters have been set successfully. An overview can be printed:")
    
    #We get the data using the parameters created above and store in gf02
    gf02 = gf02_api.get_data(params=params_gf02) 
    # We rename the variables
    gf02 = gf02.rename(columns={'KOMK': 'area', 'BRANCHEDB0710TIL127': 'industry', 'FIRMSTR': 'size', 'TID': 'year', 'INDHOLD': 'companies'})
    
    return gf02

#CONSTRUCTING SUBSETS (GF02_SIZE & GF02_INDUSTRIES)

def create_gf02_size(gf02):

    # We create an index to include all of Denmark and the total for all industries
    I = gf02.area.str.contains('All Denmark') & gf02.industry.str.contains('TOT Industry total')
    gf02_size = gf02.loc[I, :] #This cuts down the number of rows
    
    # Since we have cut down the number of rows, we want to reset the ID
    gf02_size.reset_index(inplace=True, drop=True) # We drop the old index. If the drop is not used, python will create another dataset with the old index.
    gf02_size= gf02_size.drop(['area', 'industry'], axis=1) # We drop the variables 'area' and 'industry'

    return gf02_size

def create_gf02_industries(gf02): #Same approach as above 
    
    # We create an index to include all of Denmark and total enterprises
    I = gf02.area.str.contains('All Denmark') & gf02['size'].str.contains('Total, all enterprises')
    gf02_industries = gf02.loc[I, :]
    
    gf02_industries.reset_index(inplace=True, drop=True) 
    gf02_industries = gf02_industries.drop(['area', 'size'], axis=1)
    
    return gf02_industries

# EXPLORING THE SUBDATASETS OF GF02

def table_size_overview(x):
    #We make a wide dataset for a more clear table depiction.
    #We reindex in a desired order as the program otherwise would make and order of size-names according to numbers and alphabetic order.

    table_size = pd.pivot(x, index='size', columns='year', values='companies')
    table_size = table_size.reindex(['No employed', 
                                         'Less than 10 employed', 
                                         '10-49 employees', 
                                         '50-249 employees', 
                                         '250 employees and more', 
                                         'Total, all enterprises'])
    table_size
    return table_size


def plot_size_overview(x):
    ax = (x[x['size'] != 'Total, all enterprises'].reset_index()
    .pivot(index='year', columns='size', values='companies')
    .plot(title='Number of companies in Denmark by size'))

    handles, labels = ax.get_legend_handles_labels()
    # Here we specify the desired order for the legend labels
    order_legend_size = ['No employed', 'Less than 10 employed', '10-49 employees', '50-249 employees', '250 employees and more']

    # We create a new list of handles and labels in the desired order
    handles_size = [handles[labels.index(label)] for label in order_legend_size]
    labels_size = order_legend_size

    plt.legend(title='Company Size', handles=handles_size, labels=labels_size, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def interactive_gf02_size(data):
    def plot_gf02_size(df, size): 
        I = df['size'] == size
        ax=df.loc[I,:].plot(x='year', y='companies', style='-o', legend=False)

    widgets.interact(plot_gf02_size, 
    df = widgets.fixed(data),
    size = widgets.Dropdown(description='Size', 
                        options=data['size'].unique(), 
                        value='Total, all enterprises')
);


def table_industries_overview(x):
    table_industries = pd.pivot(x, index='industry', columns='year', values='companies')
    table_industries = table_industries.reindex(['1 Agriculture, forestry and fishing', 
                                         '2 Manufacturing, mining and quarrying, and utility services', 
                                         '3 Construction', 
                                         '4 Trade and transport etc.', 
                                         '5 Information and communication', 
                                         '6 Financial and insurance',
                                         '7 Real estate',
                                         '8 Other business services',
                                         '9 Public administration, education and health',
                                         '10 Arts, entertainment and other services',
                                         '11 Activity not stated',
                                         'TOT Industry total'])
    table_industries
    return table_industries

def plot_industries_overview(x):
    ax = (x[x['industry'] != 'TOT Industry total'].reset_index()
    .pivot(index='year', columns='industry', values='companies')
    .plot(title='Number of companies in Denmark for different industries'))

    handles, labels = ax.get_legend_handles_labels()
    # Here we specify the desired order for the legend labels
    order_legend_industries = ['1 Agriculture, forestry and fishing', 
                            '2 Manufacturing, mining and quarrying, and utility services', 
                            '3 Construction', 
                            '4 Trade and transport etc.', 
                            '5 Information and communication', 
                            '6 Financial and insurance',
                            '7 Real estate',
                            '8 Other business services',
                            '9 Public administration, education and health',
                            '10 Arts, entertainment and other services',
                            '11 Activity not stated']

    # We create a new list of handles and labels in the desired order
    handles_industries = [handles[labels.index(label)] for label in order_legend_industries]
    labels_industries = order_legend_industries

    plt.legend(title='Industries', handles=handles_industries, labels=labels_industries, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


# IMPORT AND DISPLAY (DEMO4)

def import_demo4_data(x):
    #We import the data using the api
    demo4_api = DstApi(x)

    # We display the data summary and get an overview of the available values for each variable
    tabsum2 = demo4_api.tablesummary(language='en')
    display(tabsum2)

    for variable in tabsum2['variable name']:
        print(variable + ':')
        display(demo4_api.variable_levels(variable, language='en'))

    # We define a dictionary for the parameters
    params_demo4 = demo4_api._define_base_params(language='en')

    # We select all parameters for now
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

    # Finally, we load the data from dst
    demo4_raw = demo4_api.get_data(params=params_demo4)
    return params_demo4, demo4_raw

# CLEANING AND RENAMING OF DEMO4

def clean_demo4_data(x):
    # We rename the columns
    x.rename(columns={
        'REGION': 'regions',
        'BRANCHEDB0710': 'industry',
        'MÆNGDE4': 'x',
        'TID': 'year',
        'INDHOLD': 'total'
    }, inplace=True)

    # Then we select the parameters we want to examine
    I = x['regions'].str.contains('All Denmark') & x['industry'].str.contains('TOT Industry total')
    demo4_reduced = x.loc[I, :].copy()  # Make a copy to avoid SettingWithCopyWarning

    # Reset the index
    demo4_reduced.reset_index(inplace=True, drop=True)

    # We delete the column "regions" since it has been reduced to only containing "All Denmark"
    drop_this = ['regions']
    demo4_reduced.drop(drop_this, axis=1, inplace=True)

    print("Cleaning has been done successfully.")
    demo4_reduced
    return demo4_reduced

# MERGING THE DATASETS (DEMO4 AND GF02)

def merge_datasets(data1, data2):
    # We create a subset of DEMO4 including the industries
    I = data1['regions'].str.contains('All Denmark')
    data1_industries = data1.loc[I, :].copy()  # Make a copy to avoid SettingWithCopyWarning

    # We reset the index
    data1_industries.reset_index(inplace=True, drop=True)

    # We drop the 'regions' column
    drop_this = ['regions']
    data1_industries.drop(drop_this, axis=1, inplace=True)
    
    # Then we pivot the df to unpack the values of 'x' into individual columns
    data1_unpacked_merge = data1_industries.pivot(index=['industry', 'year'], columns='x', values='total').reset_index()
    data1_unpacked_merge.columns.name = None

    # We sort and reset the index of gf02_industries
    data2_industries_merge = data2.sort_values(['industry', 'year']).reset_index(drop=True)

    # We make an outer merge on industry and year
    merged_data = pd.merge(data2_industries_merge, data1_unpacked_merge, on=['industry', 'year'], how='outer')

    # Finally we rename the columns
    merged_data.rename(columns={
        'Employees (in full-time persons)': 'employees in startups',
        'Export (DKK 1000)': 'Export(DKK1000)',
        'New enterprises (number)': 'startups',
        'Terminated enterprises (number)': 'bankruptcies',
        'Turnover (DKK 1000)': 'Turnover(DKK1000)'
    }, inplace=True)

    return merged_data

# COMPANY SIZE SHARES FOR DIFFERENT INDUSTRIES (GF02)

def interactive_size_industry(data):
    #We define a function, where we filter by size and industry, so we can construct a widget with two dropdowns below.
    def plot_gf02(df, size, industry): 
        filtered_df = df[(df['size'] == size) & (df['industry'] == industry)]
        total_df = df[(df['size'] == 'Total, all enterprises') & (df['industry'] == industry)]
        
        # We merge the filtered_df with total_df on 'year' to calculate the share
        merged_df = pd.merge(filtered_df, total_df, on='year', suffixes=('', '_total'))
        
        # we calculate the share of individual size groups
        merged_df['share'] = merged_df['companies'] / merged_df['companies_total']
        
        fig, ax1 = plt.subplots()

        # We plot the number of companies
        ax1.plot(merged_df['year'], merged_df['companies'], '-o', color='blue',alpha=0.5, label='Number of companies')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of companies', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # We create a secondary y-axis to plot the share
        ax2 = ax1.twinx()
        ax2.plot(merged_df['year'], merged_df['share'], '-x', color='green', label='Share')
        ax2.set_ylabel('Share', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Finally adding the legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', bbox_to_anchor=(1.2, 0.5))
        
        plt.show()

    widgets.interact(plot_gf02, 
    df = widgets.fixed(data),
    industry=widgets.Dropdown(description='Industry', options=data['industry'].unique(), value='1 Agriculture, forestry and fishing'),
    size = widgets.Dropdown(description='Size', options=data['size'].unique(), value='Total, all enterprises')
);

# CHARACTERISTICS OF DANISH COMPANIES (DEMO4)

def interactive_demo4(data):
    def plot_demo4(df, x): 
        I = df['x'] == x
        ax = df.loc[I,:].sort_values('year').plot(x='year', y='total', style='-o', legend=False)

    widgets.interact(plot_demo4, 
        df = widgets.fixed(data),
        x = widgets.Dropdown(description='total', 
                            options=data['x'].unique(), 
                            value='Employees (in full-time persons)')
);

def plot_rates(x):
    industries_of_interest = ['1 Agriculture, forestry and fishing', 
                          '2 Manufacturing, mining and quarrying, and utility services', 
                          '3 Construction', 
                          '4 Trade and transport etc.', 
                          '5 Information and communication', 
                          '6 Financial and insurance',
                          '7 Real estate',
                          '8 Other business services',
                          '9 Public administration, education and health',
                          '10 Arts, entertainment and other services',
                          'TOT Industry total']

    # Define colors for each industry
    colors = ['Blue', 'Orange', 'Green', 'Red', 'Purple', 'Brown', 'Pink', 'Gray', 'Olive', 'Cyan', 'Black']

    # Create subplots for survival rate and startup rate
    fig, axes = plt.subplots(1, 2, figsize=(20,5))

    # Plot average survival rate
    plt.subplot(1, 2, 1)
    for i, industry in enumerate(industries_of_interest):
        avg_survival_rate = x[x['industry'] == industry]['survival_rate'].mean()
        plt.bar(i, avg_survival_rate, color=colors[i], label=industry, alpha=0.7)
    plt.xlabel('Industry')
    plt.ylabel('Average Survival Rate')
    plt.title('Average Survival Rate (2007-2021)')
    plt.gca().set_xticklabels([])

    # Plot average startup rate
    plt.subplot(1, 2, 2)
    for i, industry in enumerate(industries_of_interest):
        avg_startup_rate = x[x['industry'] == industry]['startup_rate'].mean()
        plt.bar(i, avg_startup_rate, color=colors[i], label=industry, alpha=0.7)
    plt.xlabel('Industry')
    plt.ylabel('Average Startup Rate')
    plt.title('Average Startup Rate (2007-2021)')
    plt.gca().set_xticklabels([])

    # Add a common legend
    plt.legend(title='Industry', bbox_to_anchor=(1.7, 1), loc='upper left', borderaxespad=0., fontsize='small')

    plt.tight_layout()
    plt.show()


#COMPARING SELECTED INDUSTRIES

def plot_startups(x):
    selected_industries = ['1 Agriculture, forestry and fishing', 
                       '5 Information and communication', 
                       '7 Real estate']
    filtered_df = x[x['industry'].isin(selected_industries)]

    ax = (filtered_df.reset_index()
        .pivot(index='year', columns='industry', values='startups')
        .plot(title='Number of startups in Denmark for selected industries'))

    handles, labels = ax.get_legend_handles_labels()
    order_legend_industries = selected_industries

    handles_industries = [handles[labels.index(label)] for label in order_legend_industries]
    labels_industries = order_legend_industries

    plt.legend(title='Industries', handles=handles_industries, labels=labels_industries, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_compare_companies(x):
    selected_industries = ['1 Agriculture, forestry and fishing', 
                       '5 Information and communication', 
                       '7 Real estate']
    filtered_df = x[x['industry'].isin(selected_industries)]

    ax = (filtered_df.reset_index()
        .pivot(index='year', columns='industry', values='companies')
        .plot(title='Number of companies in Denmark for selected industries'))

    handles, labels = ax.get_legend_handles_labels()
    order_legend_industries = selected_industries

    handles_industries = [handles[labels.index(label)] for label in order_legend_industries]
    labels_industries = order_legend_industries

    plt.legend(title='Industries', handles=handles_industries, labels=labels_industries, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()