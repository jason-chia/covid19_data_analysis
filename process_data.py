import numpy as np
import pandas as pd

def format_data(data):
    # rename the column names
    column_rename = ['case_id', 'age_group', 'last_reported_day', 'last_reported_month', 'death', 'p_death', 'episode_day', 
                 'episode_month', 'gender', 'hospitalization', 'p_hospitalization', 'icu', 'p_icu', 'transmission'] 
    data.columns = column_rename

    # create calculated date (yyyy-mm-dd) columns and drop the (day, month) columns
    data['last_reported_date'] = '2020-' + data.last_reported_month.map(str) + '-' + data.last_reported_day.map(str)
    data['episode_date'] = '2020-' + data.episode_month.map(str) + '-' + data.episode_day.map(str)
    data = data.drop(['last_reported_month', 'last_reported_day', 'episode_month', 'episode_day'], axis=1)

    # set episode_date to missing if not the proper date format
    data["episode_date"].replace({"2020-99-99": ""}, inplace=True)

    # convert data columns from string to datetime dateype
    data['episode_date'] = pd.to_datetime(data['episode_date'])
    data['last_reported_date'] = pd.to_datetime(data['last_reported_date'])

    # reorder the column variables
    column_reorder = ['case_id', 'episode_date', 'last_reported_date', 'age_group', 'gender', 'transmission', 
                  'hospitalization', 'p_hospitalization', 'icu', 'p_icu', 'death', 'p_death'] 
    data = data[column_reorder]

    return data;

def recode_variable(data):
    # recode all the binary categorical variables (excluding age_group) to {0,1,9}
    # for all except gender, 0 - 'No', 1 - 'Yes', 9 -'Unknown'
    # for gender, 0 - 'Female', 1 - 'Male', 9 - 'Unknown'
    column_recode = data.columns.tolist()[4:12] 
    data[column_recode] = data[column_recode].replace({2: 0})
    data[['gender', 'transmission']] = data[['gender', 'transmission']].replace({3: 9})

    # recode the age_group variable by taking mean of each age group
    # age_group 0-19: code as 10
    # age_group 20-29: code as 25
    # age_group 30-39: code as 35
    # age_group 40-49: code as 45
    # age_group 50-59: code as 55
    # age_group 60-69: code as 65
    # age_group 70-79: code as 75
    # age_group 80+: code as 85
    # age_group Unknown: code as 99
    data[['age_group']] = data[['age_group']].replace({1: 10})
    data[['age_group']] = data[['age_group']].replace({2: 25})
    data[['age_group']] = data[['age_group']].replace({3: 35})
    data[['age_group']] = data[['age_group']].replace({4: 45})
    data[['age_group']] = data[['age_group']].replace({5: 55})
    data[['age_group']] = data[['age_group']].replace({6: 65})
    data[['age_group']] = data[['age_group']].replace({7: 75})
    data[['age_group']] = data[['age_group']].replace({8: 85})

    return data

def get_variable(data):
    # output variable description
    name = data.columns.tolist()
    description = ['Case Identifier ID', 'Date Case was Confirmed', 'Date Case was Last Updated', 'Age Group', 'Gender',
               'Mode of Transmission', 'Hospitalization Status','Hospitalization Previous Status','Intensive Care Unit Status',
               'Intensive Care Unit Previous Status', 'Death Status', 'Death Previous Status']
    variable = pd.DataFrame(data=[name,description]).T
    variable.columns = ['Variable Name','Variable Description']
    return variable