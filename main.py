# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 19:51:39 2019

@author: Shyam
"""
#Training of model, and saving the Gridsearch CV object.
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import json
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, ShuffleSplit
from scipy import stats
from sklearn.externals import joblib
#argparse
from opts import parser
args = parser.parse_args()

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

def convert_to_categorical(data, columns):
    for each in columns:
        data[each] = data[each].astype('category')

def preprocess_data(data, train = args.train):
    data.set_index('instant', inplace=True)
    data.drop(['dteday','casual','registered'], axis = 1, inplace = True)
    if train:
        data.drop(data[data['weathersit'] == 4].index, inplace = True)
  
def make_preparation_pipeline():
    num_attribs = ['temp', 'atemp', 'hum', 'windspeed']
    cat_attribs = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']
    
    num_pipeline = Pipeline([
                    ('selector', DataFrameSelector(num_attribs)),
                    ('imputer', SimpleImputer(strategy= 'median')),
                    #('std_scaler', StandardScaler()),
    ])
    
    cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(cat_attribs)),
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(sparse  = False, handle_unknown = "ignore"))
    ])
    
    
    #Combining both the numerical and categorical features pipeline
    preparation_pipeline  = FeatureUnion(transformer_list = [('num_pipeline', num_pipeline),
                                                     ('cat_pipeline', cat_pipeline)
                                                     ])
    return preparation_pipeline    

def save_histogram(y_actual, y_predicted, train = args.train):
    train_errors = y_predicted - y_actual
    plt.hist(train_errors, bins = 100)    
    plt.xlabel('Error')
    plt.ylabel('Count')
    if train:
        plt.title('Train Error distribution')
        plt.savefig('Train_error_histogram.jpg')
    else:
        plt.title('Test Error distribution')
        plt.savefig('Test_error_histogram.jpg')        

def save_scatterplot(y_actual, y_predicted, train = args.train):
    plt.scatter(x = y_actual, y = y_predicted)

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    #plt.show()
    if train:
        plt.title('Train data scatterplot')
        plt.savefig('Train_scatter.jpg')
    else:
        plt.title('Test data scatterplot')
        plt.savefig('Test_scatter.jpg')
        
        
        
"""
If single regressor is None, you will get the result of Grid search for all the models.
If you want to pass different parameters in the parameter grid, you can do so by parameters_grid_passed, which
is a dictionary.
"""
      
def grid_search_multi_models(X_train, y_train, single_regressor = None, parameters_grid_passed = None):
    global preparation_pipeline
    regressors = [
        ('Linear Regression', LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)),
        ('Lasso Regression', Lasso(fit_intercept=True, normalize=False, copy_X=True, random_state=1)),
        ('Gradient Boost Regressor', GradientBoostingRegressor(random_state = 1)),
        ('Random Forest Regressor', RandomForestRegressor(random_state = 1)),
        ('Neural Networks', MLPRegressor(random_state = 50, activation='relu', max_iter = 100)),
    ]
    
    if single_regressor:
        regressors = single_regressor

    params_grid = {
        'Linear Regression': {},
        'Lasso Regression': {
            'alpha': [0.3,0.5,1.0],
        },
        'Gradient Boost Regressor': {
            'learning_rate':[0.1],
            'n_estimators':[50,70,100],
            'max_depth':[5,10],
        },
        
        'Random Forest Regressor' : {     
            'n_estimators': [50,70,100],
            'max_depth': [5,10],
        },
        
        'Neural Networks': {
            'hidden_layer_sizes': [(10,10),(10,10,10)],
        },
        
    }
    if parameters_grid_passed:
        for each in parameters_grid_passed:
            params_grid[each] = parameters_grid_passed[each]
    
    results = pd.DataFrame(columns = ["Best tuned model", "Train Accuracy","Train MAE", "Best hyper parameters"])
    for (name, regressor) in regressors:
        
        parameters = params_grid[name]

        #preparing th pipeline for estimator

        preparation_pipeline_with_regressor = Pipeline([
        ("preparation", preparation_pipeline),
        ("regressor", regressor)
        ])

        # Perform the grid search for best parameters

        hyper_params = {}
        for params in parameters.keys():
            hyp_p = 'regressor__'+str(params)
            hyper_params[hyp_p] = parameters[params] 
        
        print("Performing Grid Search for", str(name))
        grid_search_clf = GridSearchCV(preparation_pipeline_with_regressor, hyper_params, scoring='neg_mean_absolute_error', n_jobs = -1, cv=5, verbose = 2)
        grid_search_clf.fit(X_train, y_train)
                        
        # Store the results
        best_train_accuracy = grid_search_clf.best_estimator_.score(X_train, y_train)
        y_train_predicted = grid_search_clf.best_estimator_.predict(X_train)
        train_mae = mean_absolute_error(np.array(y_train), y_train_predicted)
        best_parameters = grid_search_clf.best_estimator_.get_params()
        param_dummy = []
        for param_name in sorted(hyper_params.keys()):
            param_dummy.append((param_name, best_parameters[param_name]))
        results.loc[len(results)] = [name, best_train_accuracy,train_mae, json.dumps(param_dummy)]
    print('Results of model')
    pd.set_option('display.max_colwidth', -1)
    print(results)    
    if single_regressor:
        save_scatterplot(y_train, y_train_predicted, train = args.train)
        save_histogram(y_train, y_train_predicted, train = args.train)
        return grid_search_clf


if __name__ == "__main__":
    filename = args.filename
    data = pd.read_csv(filename)
    preprocess_data(data, train = args.train)
    categorical_columns = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']
    convert_to_categorical(data, categorical_columns)
    preparation_pipeline = make_preparation_pipeline()
    if args.train:
        X_train = data.loc[:,data.columns != 'cnt']
        y_train = data['cnt']        
        single_regressor =  [('Gradient Boost Regressor', GradientBoostingRegressor(random_state = 1)),]
        parameters_grid_passed = {'Gradient Boost Regressor': {
                    'learning_rate':[0.1],
                    'n_estimators':[200],
                    'max_depth':[10],
                }}
        
        grid_search_clf = grid_search_multi_models(X_train, y_train,single_regressor, parameters_grid_passed)
        joblib.dump(grid_search_clf.best_estimator_, 'model.pkl', compress = 1)
    else:
        X_test = data.loc[:,data.columns != 'cnt']
        print('Loading pickle file')
        model_loaded = joblib.load('model.pkl')
        #print(model_loaded)
        print('Model loaded')
        y_test_predicted = model_loaded.predict(X_test)
        data['Y_predicted'] = y_test_predicted
        data.to_csv('TestDataPredictions.csv', index = True)
        if 'cnt' in data.columns:
            y_test = data['cnt']
            mae_test = mean_absolute_error(np.array(y_test), y_test_predicted)
            test_accuracy = r2_score(y_test, y_test_predicted)
            results = pd.DataFrame(columns = ["Best tuned model", "Test Accuracy","Test MAE"])
            results.loc[len(results)] = ['Gradient Boost Regressor', test_accuracy,mae_test]
            pd.set_option('display.max_colwidth', -1)
            print(results)
            save_scatterplot(y_test, y_test_predicted, train = args.train)
            save_histogram(y_test, y_test_predicted, train = args.train)