import process_data as process
import perform_analysis as analyze
import generate_graph as graph
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# download data from https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310076601 on May 8, 2020
# load statscan covid19 data and format the VALUE column (N = 409,123)
covid19 = pd.read_csv('https://raw.githubusercontent.com/jason-chia/covid19_data_analysis/master/13100766.csv', low_memory=False)
covid19['VALUE'] = covid19['VALUE'].fillna(9)
covid19['VALUE'] = covid19['VALUE'].astype(int)
print('\n')
print('Raw dataset Downloaded from Statistics Canada on May 8, 2020 - 13 Records per COVID-19 Infected Person (N = 409,123)')
print('\n')
print(covid19.head(15))
print('\n')

# transponse the data using the pivot option (N = 31,471)
columns = covid19.columns.tolist()
data = covid19.pivot(index=columns[0], columns=columns[1], values = columns[2]).reset_index().rename_axis(None, axis=1)

# format the dataset including renaming column names, concatenating dates, and reordering variables 
data = process.format_data(data)

# recode the variables
data = process.recode_variable(data)
print('\n')
print('One Row per COVID-19 Infected Person after Data Transposing, Formatting, and Recoding (N = 31,471)')
print('\n')
print(data[['case_id', 'episode_date', 'last_reported_date', 'age_group', 'gender', 'transmission']].head(5))
print('\n')
print(data[['case_id', 'hospitalization', 'p_hospitalization', 'icu', 'p_icu', 'death', 'p_death']].head(5))
print('\n')

# get variable name and description of the dataset
variable = process.get_variable(data)
print('\n')
print('Variable Name and Description')
print('\n')
print(tabulate(variable, variable.columns, tablefmt="github"))
print('\n')

# define cohort by excluding missing episode_date (N = 30,182)
# define episode_days as the number of days from the earliest date which is Jan 15, 2020
cohort = data[data.episode_date.notnull()]
episode_days = cohort.episode_date - min(cohort['episode_date'].unique())
episode_days = episode_days.astype('timedelta64[D]').astype(int).tolist()
cohort.insert(2, "episode_days", episode_days, True) 

# display 2-way frequency tables between (age_group, gender, transmission) and death
freq_tables = analyze.generate_frequency(cohort)
varname = ['Age Group', 'Gender', 'Transmission']
for i in range(0, len(freq_tables)):
    print('\n')    
    print('2-Way Frequency Table: ' + varname[i] +' vs COVID-19 Death Status as of May 8, 2020 (Excluding Cases with Missing Confirmation Date)\n')
    print(freq_tables[i] + '\n')
    print('\n')
    
# generate a histgram of the daily number of new cases, stratified by their death status as of May 8, 2020
graph.display_daily_num(cohort)

# generate histogram of the density and overlay a curve on top of the density
graph.display_daily_density(cohort)

# generate a histgram of the cumulative number of cases, stratified by their death status as of May 8, 2020
graph.display_cumulative_num(cohort)

# generate histogram of the cumulative density and overlay a curve on top of the cumulative density
graph.display_cumulative_density(cohort)

# define cohort2 by excluding missing age_group, gender, transmission, and death from cohort (N = 13,214)
# use cohort2 to model the death status (response) by age_group, gender, and transmission (predictors)
# split the dataset into training (2/3) and validation (1/3)
cohort2 = cohort[(cohort.age_group != 99) & (cohort.gender !=9) & (cohort.transmission != 9) & (cohort.death != 9)]
x = cohort2[['age_group', 'gender', 'transmission']]
y = cohort2['death'] 
x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=1/3, random_state=1)

# perform chi-square test to see which predictors are dependent on death using the training dataset at alpha = 0.05
# predictors: age_group (8 levels), gender (2 levels), transmission (2 levels)
# response: death (2 levels)
chi2_test = analyze.perform_chi2_test(x_train, y_train, 0.05)
print('\n')    
print('Chi-Sqaure Test between (Age Group, Gender, Mode of Transmission) and COVID-19 Death Status Using the Training Dataset\n')
print(tabulate(chi2_test, chi2_test.columns, tablefmt="github", floatfmt=".2f"))
print('\n')

# perform one additional chi-square test betweeen each age_group vs death (total 8) using the training dataset at alpha=0.05
age_group_dummies = pd.get_dummies(x_train['age_group'])
age_group_dummies.columns = ['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
chi2_age_group = analyze.perform_chi2_test(age_group_dummies, y_train, 0.05/cohort2['age_group'].nunique())
print('\n')    
print('Chi-Sqaure Test between Each Age Group vs COVID-19 Death Status Using the Training Dataset\n')
print(tabulate(chi2_age_group, chi2_age_group.columns, tablefmt="github", floatfmt=".2f"))
print('\n')

# fit models to see what attributes of the COVID-19 infected individual is prone to death
# predictors: age_group (8 levels), gender (2 levels), transmission (2 levels)
# response: death (2 levels)
# treat age_group as ordinary/continuous (treating age_group as categorical previously produces the same prediction accuracy)
# define 6 models (chi-square test for gender is not significant in the dataset downloaded on April 30, 2020)
# 1. logistic regression with gender
# 2. logistic regression without gender
# 3. support vector machine with gender
# 4. support vector machine without gender
# 5. naive bayes classifier with gender
# 6. naive bayes classifier without gender
models = []
models.append((LogisticRegression(),['age_group','gender','transmission'], 'Logistic Regression with Gender'))
models.append((LogisticRegression(),['age_group','transmission'], 'Logistic Regression without Gender'))
models.append((SVC(gamma='auto',kernel='rbf'), ['age_group','gender','transmission'], 'Support Vector Machine with Gender'))
models.append((SVC(gamma='auto',kernel='rbf'), ['age_group','transmission'], 'Support Vector Machine without Gender'))
models.append((GaussianNB(), ['age_group','gender','transmission'], 'Naive Bayes Classifier with Gender'))
models.append((GaussianNB(), ['age_group','transmission'], 'Naive Bayes Classifier without Gender'))

# fit the model using the training dataset
selection, classification_tables = analyze.fit_mdoel(models, x_train, y_train, x_validate, y_validate)

# output prediction accuracy for each model using the validation dataset
print('\n')    
print('Prediction Accuracy of Each Model on the Validation Dataset\n')
print(tabulate(selection, selection.columns, tablefmt="github", floatfmt=".4f"))
print('\n')

# output the confusion matrix for each model using the validation dataset
for i in range(0, len(models)):
    print('\n')    
    print(models[i][2] + ' on the Validation Dataset\n')
    print(classification_tables[i])
    print('\n') 

# show logistic regression without gender in greater detail: log(P(death=1)) ~ age_group + transmission
# exclude gender from the model as the model seems to predict the number of actual death better
parameters = analyze.fit_logistic_regression(x_train.drop('gender', axis = 1), y_train, x_validate.drop('gender', axis = 1), y_validate)
print('\n')    
print('Parameter Estimate of the Logistic Regression Model without Gender Using the Training Dataset\n')
print(tabulate(parameters, parameters.columns, tablefmt="github", floatfmt=".4f"))
print('\n') 

# plot the Sigmoid function using the parameter coefficient estimate from the logistic regression without gender model
graph.display_sigmoid(parameters)