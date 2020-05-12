import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate

def generate_frequency(cohort):
    varname = ['age_group', 'gender', 'transmission']
    rowlabels = [['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+', 'Unknown', 'Total' ], ['Female', 'Male', 
              'Unknown', 'Total'], ['Community Exposure', 'Travel Exposure', 'Unknown', 'Total']]
    columnlabels = ['Death No', 'Death Yes', 'Death Unknown', 'Total']

    # store all the frequency tables in a list and create row and column labels
    freq_tables = []
    for i in varname:
        freq_tables.append(pd.crosstab(cohort[i], cohort['death']))

    # output the frequency tables in github format (make them pretty) using the tabulate library
    freq_tables_github = []
    for i in range(0, len(freq_tables)):
        values = freq_tables[i].values.T.tolist()
        column_total = freq_tables[i].sum().tolist()
        for j in range(0, len(values)):
            values[j] = values[j] + [column_total[j]]
        row_total = freq_tables[i].sum(axis=1).tolist() + [freq_tables[i].sum().sum()]
        table = np.array([rowlabels[i]] + values + [row_total]).T
        freq_tables_github.append(tabulate(table, columnlabels, tablefmt="github"))
        
    return freq_tables_github

def perform_chi2_test(x, y, alpha):
    # perform chi-square testing between the predictors and the reponse variables using the training dataset at alpha
    chi2_test = []
    for i in x:
        temp = []
        temp.append(i)
        chi2_stat = chi2_contingency(pd.crosstab(x[i], y))
        temp.append('%.2f' % chi2_stat[0])
        temp.append('%.2f' % chi2_stat[1])
        if chi2_stat[1] < alpha:
            temp.append('Yes')
        else:
            temp.append('No')
        chi2_test.append(temp)
    chi2_test = pd.DataFrame(data = chi2_test)
    chi2_test.columns = ['Predictor Variable', 'Chi-Square', 'p-value', 'significance']
    
    return chi2_test

def fit_mdoel(models, x_train, y_train, x_validate, y_validate):
    # run each model and store model results
    selection = []
    classification = []
    for i in models:
        # build the model
        model = i[0]
        column_select = i[1]
        # fit the training dataset
        model.fit(x_train[column_select], y_train)
        # make prediction on the validation dataset
        predictions = model.predict(x_validate[column_select])
        # get prediction accuracy
        accuracy = accuracy_score(y_validate, predictions)
        # get the confusion matrix
        classification.append(pd.DataFrame(data=confusion_matrix(y_validate, predictions)))
        temp = []
        temp.append(i[2])
        temp.append(accuracy.astype(str))
        selection.append(temp)
    selection = pd.DataFrame(data = selection)
    selection.columns = ['Model','Prediction Accuracy']

    # Output the confusion matrix for each model using the validation dataset
    rowlabels = ['Actual Death No', 'Actual Death Yes', 'Total']
    columnlabels = ['Predicted Death No', 'Predicted Death Yes', 'Total']
    classification_tables = []
    for i in range(0, len(classification)):
        values = classification[i].values.T.tolist()
        column_total = classification[i].sum().tolist()
        for j in range(0, len(values)):
            values[j] = values[j] + [column_total[j]]
        row_total = classification[i].sum(axis=1).tolist() + [classification[i].sum().sum()]
        table = np.array([rowlabels] + values + [row_total]).T
        classification_tables.append(tabulate(table, columnlabels, tablefmt="github"))
        
    return selection, classification_tables

def fit_logistic_regression(x_train, y_train, x_validate, y_validate):
    # fit to a logistic regression model
    log_model = LogisticRegression()
    log_model.fit(x_train, y_train)

    # get coefficient of the paramter estimate including the intercept term
    coefficient = log_model.intercept_.tolist() + log_model.coef_.tolist()[0]

    # compute standard error of the parameter estimate
    # https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients
    y_train_prob = log_model.predict_proba(x_train)
    x_design = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    v = np.diagflat(np.product(y_train_prob, axis=1))
    covLogit = np.linalg.inv(x_design.T @ v @ x_design)
    standard_error = np.sqrt(np.diag(covLogit)).tolist()
    
    # store coefficient, standard error, and 95% confidence intervals in a DataFrame
    parameters = pd.DataFrame(data = [['intercept'] + x_train.columns.tolist(), coefficient, standard_error]).T
    parameters.columns = ['Variable','Coefficient','Standard Error']
    parameters['95% Lower'] = parameters.Coefficient - 1.96 * parameters['Standard Error']
    parameters['95% Upper'] = parameters.Coefficient + 1.96 * parameters['Standard Error']

    return parameters