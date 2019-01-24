""" Module Import """
# scipy
import scipy as sc
print('scipy: %s' % sc.__version__)
# numpy
import numpy as np
print('numpy: %s' % np.__version__)
# pandas
import pandas as pd
print('pandas: %s' % pd.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn.metrics import r2_score
import sklearn as sk
print('sklearn: %s' % sk.__version__)
# math
import math
import csv
import os.path
import time

""" Config Setup """
iteration = 0
max_lag_param = 13
test_harness_param = (.5,.6,.7,.8,.9)
max_features_param=('sqrt','log2', None)
max_depth=(3, 6, None)
n_estimators = 500
parallel_degree = -1

""" Feature Engineering """

# Test Multiple Time Splits (Lag)
for lag in range(1,max_lag_param+1):
    t0 = time.time()
    shifted_df = series_to_supervised(df, lag, 1)

    # Seperate labels from features
    y_df_column_names = shifted_df.columns[len(df.columns):len(df.columns) + len(y_label)]
    y_df = shifted_df[y_df_column_names]
    X_df = shifted_df.drop(columns=y_df_column_names)

    # Delete middle timesteps
    X_df = remove_n_time_steps(data=X_df, n_in=lag)

    # Test Multiple Train/Validation Splits
    for test_split in test_harness_param:
        X_train, X_validate, y_train, y_validate = train_test_split(X_df, y_df, test_size=test_split)
        X_train = X_train.values
        y_train = y_train.values
        X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=.5)
        X_validate = X_validate.values
        y_validate = y_validate.values

        # Train Multiple Regression Forest Models using various estimators
        for max_features in max_features_param:
            for depth in max_depth:
                model = RandomForest(mode='regression',
                                     n_estimators=n_estimators,
                                     parallel_degree=parallel_degree,
                                     max_depth=max_depth,
                                     max_features=max_features)
                model.fit_model(X_train=X_train,
                                y_train=y_train)
                rmse = model.predict_and_evaluate(X=X_validate,
                                                  y=y_validate,
                                                  y_labels=y_label,
                                                  plot=False)
                t1 = time.time()
                time_total = t1 - t0
                RandomForest.write_results_to_disk(path="time_series_random_forest_regression_results.csv",
                                                   iteration=iteration,
                                                   lag=lag,
                                                   test_split=test_split,
                                                   estimator=n_estimators,
                                                   score=rmse,
                                                   time_train=time_total)
                iteration += 1