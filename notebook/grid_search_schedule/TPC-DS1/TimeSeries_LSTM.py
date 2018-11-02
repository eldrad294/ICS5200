#
# Module Imports
# scipy
import scipy as sc
print('scipy: %s' % sc.__version__)
# numpy
import numpy as np
print('numpy: %s' % np.__version__)
# matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
# pandas
import pandas as pd
print('pandas: %s' % pd.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, roc_curve, roc_auc_score
import sklearn as sk
print('sklearn: %s' % sk.__version__)
# theano
import theano
print('theano: %s' % theano.__version__)
# tensorflow
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
# keras
import keras as ke
print('keras: %s' % ke.__version__)
# math
import math
import csv
import os.path
import time
#
# Experiment Config
tpcds='TPCDS1' # Schema upon which to operate test
lag=1 # Time Series shift / Lag Step. Each lag value equates to 1 minute. Cannot be less than 1
sub_sample_start=350 # Denotes frist 0..n samples (Used for plotting purposes)
y_label = ['CPU_TIME_DELTA','OPTIMIZER_COST','EXECUTIONS_DELTA','ELAPSED_TIME_DELTA'] # Denotes which label to use for time series experiments
#
# LSTM Network Structure eg:
#Layers -> [Input, Hidden, Hidden, Output]
#
# Open Data
rep_hist_snapshot_path = 'C:/Users/gabriel.sammut/University/Data_ICS5200/Schedule/' + tpcds + '/v1/rep_hist_snapshot.csv'
rep_hist_sysmetric_summary_path = 'C:/Users/gabriel.sammut/University/Data_ICS5200/Schedule/' + tpcds + '/v1/rep_hist_sysmetric_summary.csv'
rep_hist_sysstat_path = 'C:/Users/gabriel.sammut/University/Data_ICS5200/Schedule/' + tpcds + '/v1/rep_hist_sysstat.csv'
# rep_hist_snapshot_path = 'D:/Projects/Datagenerated_ICS5200/Schedule/' + tpcds + '/v2/rep_hist_snapshot.csv'
# rep_hist_sysmetric_summary_path = 'D:/Projects/Datagenerated_ICS5200/Schedule/' + tpcds + '/v2/rep_hist_sysmetric_summary.csv'
# rep_hist_sysstat_path = 'D:/Projects/Datagenerated_ICS5200/Schedule/' + tpcds + '/v2/rep_hist_sysstat.csv'
#
rep_hist_snapshot_df = pd.read_csv(rep_hist_snapshot_path)
rep_hist_sysmetric_summary_df = pd.read_csv(rep_hist_sysmetric_summary_path)
rep_hist_sysstat_df = pd.read_csv(rep_hist_sysstat_path)
#
def prettify_header(headers):
    """
    Cleans header list from unwated character strings
    """
    header_list = []
    [header_list.append(header.replace("(","").replace(")","").replace("'","").replace(",","")) for header in headers]
    return header_list
#
rep_hist_snapshot_df.columns = prettify_header(rep_hist_snapshot_df.columns.values)
rep_hist_sysmetric_summary_df.columns = prettify_header(rep_hist_sysmetric_summary_df.columns.values)
rep_hist_sysstat_df.columns = prettify_header(rep_hist_sysstat_df.columns.values)
#
# Table Pivoting + Merging
print('Header Lengths [Before Pivot]')
print('REP_HIST_SNAPSHOT: ' + str(len(rep_hist_snapshot_df.columns)))
print('REP_HIST_SYSMETRIC_SUMMARY: ' + str(len(rep_hist_sysmetric_summary_df.columns)))
print('REP_HIST_SYSSTAT: ' + str(len(rep_hist_sysstat_df.columns)))
#
# Table REP_HIST_SYSMETRIC_SUMMARY
rep_hist_sysmetric_summary_df = rep_hist_sysmetric_summary_df.pivot(index='SNAP_ID', columns='METRIC_NAME', values='AVERAGE')
rep_hist_sysmetric_summary_df.reset_index(inplace=True)
rep_hist_sysmetric_summary_df[['SNAP_ID']] = rep_hist_sysmetric_summary_df[['SNAP_ID']].astype(int)
#rep_hist_sysmetric_summary_df = rep_hist_sysstat_df.groupby(['SNAP_ID']).sum()
rep_hist_sysmetric_summary_df.reset_index(inplace=True)
rep_hist_sysmetric_summary_df.sort_values(by=['SNAP_ID'],inplace=True,ascending=True)
#
# Table REP_HIST_SYSSTAT
rep_hist_sysstat_df = rep_hist_sysstat_df.pivot(index='SNAP_ID', columns='STAT_NAME', values='VALUE')
rep_hist_sysstat_df.reset_index(inplace=True)
rep_hist_sysstat_df[['SNAP_ID']] = rep_hist_sysstat_df[['SNAP_ID']].astype(int)
#rep_hist_sysstat_df = rep_hist_sysstat_df.groupby(['SNAP_ID']).sum()
rep_hist_sysstat_df.reset_index(inplace=True)
rep_hist_sysstat_df.sort_values(by=['SNAP_ID'],inplace=True,ascending=True)
#
rep_hist_sysmetric_summary_df.rename(str.upper, inplace=True, axis='columns')
rep_hist_sysstat_df.rename(str.upper, inplace=True, axis='columns')
#
# Group By Values by SNAP_ID , sum all metrics (for table REP_HIST_SNAPSHOT)
rep_hist_snapshot_df = rep_hist_snapshot_df.groupby(['SNAP_ID','DBID','INSTANCE_NUMBER']).sum()
rep_hist_snapshot_df.reset_index(inplace=True)
#
print('\nHeader Lengths [After Pivot]')
print('REP_HIST_SNAPSHOT: ' + str(len(rep_hist_snapshot_df.columns)))
print('REP_HIST_SYSMETRIC_SUMMARY: ' + str(len(rep_hist_sysmetric_summary_df.columns)))
print('REP_HIST_SYSSTAT: ' + str(len(rep_hist_sysstat_df.columns)))
#
# DF Shape
print('\nDataframe shapes:\nTable [REP_HIST_SNAPSHOT] - ' + str(rep_hist_snapshot_df.shape))
print('Table [REP_HIST_SYSMETRIC_SUMMARY] - ' + str(rep_hist_sysmetric_summary_df.shape))
print('Table [REP_HIST_SYSSTAT] - ' + str(rep_hist_sysstat_df.shape))
#
# Dealing with Empty Values
def get_na_columns(df, headers):
    """
    Return columns which consist of NAN values
    """
    na_list = []
    for head in headers:
        if df[head].isnull().values.any():
            na_list.append(head)
    return na_list
#
print('N/A Columns\n')
print('\n REP_HIST_SNAPSHOT Features ' + str(len(rep_hist_snapshot_df.columns)) + ': ' + str(get_na_columns(df=rep_hist_snapshot_df,headers=rep_hist_snapshot_df.columns)) + "\n")
print('REP_HIST_SYSMETRIC_SUMMARY Features ' + str(len(rep_hist_sysmetric_summary_df.columns)) + ': ' + str(get_na_columns(df=rep_hist_sysmetric_summary_df,headers=rep_hist_sysmetric_summary_df.columns)) + "\n")
print('REP_HIST_SYSSTAT Features ' + str(len(rep_hist_sysstat_df.columns)) + ': ' + str(get_na_columns(df=rep_hist_sysstat_df,headers=rep_hist_sysstat_df.columns)) + "\n")
#
def fill_na(df):
    """
    Replaces NA columns with 0s
    """
    return df.fillna(0)
#
# Populating NaN values with amount '0'
rep_hist_snapshot_df = fill_na(df=rep_hist_snapshot_df)
rep_hist_sysmetric_summary_df = fill_na(df=rep_hist_sysmetric_summary_df)
rep_hist_sysstat_df = fill_na(df=rep_hist_sysstat_df)
#
# Merging Frames
df = pd.merge(rep_hist_snapshot_df, rep_hist_sysmetric_summary_df,how='inner',on ='SNAP_ID')
df = pd.merge(df, rep_hist_sysstat_df,how='inner',on ='SNAP_ID')
print(df.shape)
#
# Data Ordering
df.sort_values(by=['SNAP_ID'], ascending=True, inplace=True)
#
# Floating point precision conversion
df.astype('float32', inplace=True)
df = np.round(df, 3) # rounds to 3 dp
#
# Feature Seelction
def drop_flatline_columns(df):
    columns = df.columns
    flatline_features = []
    for i in range(len(columns)):
        try:
            std = df[columns[i]].std()
            if std == 0:
                flatline_features.append(columns[i])
        except:
            pass
    #
    #print('Features which are considered flatline:\n')
    #for col in flatline_features:
    #    print(col)
    print('\nShape before changes: [' + str(df.shape) + ']')
    df = df.drop(columns=flatline_features)
    print('Shape after changes: [' + str(df.shape) + ']')
    print('Dropped a total [' + str(len(flatline_features)) + ']')
    return df
#
print('Before column drop:')
print(df.shape)
df = drop_flatline_columns(df=df)
print('\nAfter flatline column drop:')
print(df.shape)
dropped_columns_df = [ 'PLAN_HASH_VALUE',
                       'OPTIMIZER_ENV_HASH_VALUE',
                       'LOADED_VERSIONS',
                       'VERSION_COUNT',
                       'PARSING_SCHEMA_ID',
                       'PARSING_USER_ID']
df.drop(columns=dropped_columns_df, inplace=True)
print('\nAfter additional column drop:')
print(df.shape)
#
# Outlier Detection
def get_outliers_quartile(df=None, headers=None):
    """
    Detect and return which rows are considered outliers within the dataset, determined by :quartile_limit (99%)
    """
    outlier_rows = [] # This list of lists consists of elements of the following notation [column,rowid]
    for header in headers:
        outlier_count = 0
        try:
            q25, q75 = np.percentile(df[header], 25), np.percentile(df[header], 75)
            iqr = q75 - q25
            cut_off = iqr * .6 # This values needs to remain as it. It was found to be a good value so as to capture the relavent outlier data
            lower, upper = q25 - cut_off, q75 + cut_off
            #
            series_row = (df[df[header] > upper].index)
            outlier_count += len(list(np.array(series_row)))
            for id in list(np.array(series_row)):
                outlier_rows.append([header,id])
            #
            series_row = (df[df[header] < lower].index)
            outlier_count += len(list(np.array(series_row)))
            for id in list(np.array(series_row)):
                outlier_rows.append([header,id])
            print(header + ' - [' + str(outlier_count) + '] outliers')
        except Exception as e:
            print(str(e))
    #
    unique_outlier_rows = []
    for col, rowid in outlier_rows:
        unique_outlier_rows.append([col,rowid])
    return unique_outlier_rows
#
#Printing outliers to screen
outliers = get_outliers_quartile(df=df,
                                 headers=y_label)
print('Total Outliers: [' + str(len(outliers)) + ']\n')
for label in y_label:
    min_val = df[label].min()
    max_val = df[label].max()
    mean_val = df[label].mean()
    std_val = df[label].std()
    print('Label[' + label + '] - Min[' + str(min_val) + '] - Max[' + str(max_val) + '] - Mean[' + str(mean_val) + '] - Std[' + str(std_val) + ']')
print('\n---------------------------------------------\n')
for i in range(len(outliers)):
    print('Header [' + str(outliers[i][0]) + '] - Location [' + str(outliers[i][1]) + '] - Value [' + str(df.iloc[outliers[i][1]][outliers[i][0]]) + ']')
#
# Editing of Outliers
def edit_outliers(df=None, headers=None):
    """
    This method uses the interquartile method to edit all outliers to std.
    """
    outliers = get_outliers_quartile(df=df,
                                     headers=y_label)
    for label in y_label:
        min_val = df[label].min()
        max_val = df[label].max()
        mean_val = df[label].mean()
        std_val = df[label].std()
        #
        for i in range(len(outliers)):
            if label == outliers[i][0]:
                df[label].iloc[outliers[i][1]] = mean_val + std_val
                # print('Header [' + str(outliers[i][0]) + '] - Location [' + str(outliers[i][1]) + '] - Value [' + str(df.iloc[outliers[i][1]][outliers[i][0]]) + ']')
    return df
#
print("DF with outliers: " + str(df.shape))
df = edit_outliers(df=df,
                   headers=y_label)
print("DF with edited outliers: " + str(df.shape))
#
# Data Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
df_normalized_values = scaler.fit_transform(df.values)
#
df = pd.DataFrame(data=df_normalized_values, columns=df.columns)
del df_normalized_values
print(str(df.shape))
print(df.head())
#
# Rearranging Labels
y_label.append('SNAP_ID')
y_df = df[y_label]
del y_label[-1]
df.drop(columns=y_label, inplace=True)
print("Label " + str(y_label) + " shape: " + str(y_df.shape))
print("Feature matrix shape: " + str(df.shape))
#
# Merging labels and features in respective order
df = pd.merge(y_df,df,on='SNAP_ID',sort=False,left_on=None, right_on=None)
print('Merged Labels + Vectors: ' + str(df.shape))
print(df.head())
#
# Time Series Shifting
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = data
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
#
def remove_n_time_steps(data, n_in=1):
    if n_in == 0:
        return data
    df = data
    headers = df.columns
    dropped_headers = []
    for header in headers:
        if "(t)" in header:
            dropped_headers.append(header)
    #
    for i in range(1,n_in):
        for header in headers:
            if "(t-"+str(i)+")" in header:
                dropped_headers.append(str(header))
    #
    return df.drop(dropped_headers, axis=1)
#
# Frame as supervised learning set
shifted_df = series_to_supervised(df, lag, 1)
#
# Seperate labels from features
y_df_column_names = shifted_df.columns[len(df.columns):len(df.columns) + len(y_label)]
y_df = shifted_df[y_df_column_names]
X_df = shifted_df.drop(columns=y_df_column_names)
print('\n-------------\nFeatures')
print(X_df.columns)
print(X_df.shape)
print('\n-------------\nVectors')
print(y_df.columns)
print(y_df.shape)
#
# Delete middle timesteps
X_df = remove_n_time_steps(data=X_df, n_in=lag)
print('\n-------------\nFeatures After Time Shift')
print(X_df.columns)
print(X_df.shape)
#
# Bucket Function
def discretize_value(amount):
    """
    Assumes that amount is decimal value. If so, return 1st value after decimal.
    """
    amount = float(amount)
    if amount < 0:
        amount = 0.01
    if amount > 1:
        amount = 0.99
    amount = str(amount)
    amount = amount.split('.')
    amount = amount[1][0]
    return float(amount)
#print(discretize_value(amount=0.26))
#
# LSTM Class
class LSTM:
    """
    Long Short Term Memory Neural Net Class
    """

    #
    def __init__(self, X_train, y_train, layers, mode='regression', optimizer='sgd'):
        """
        Initiating the class creates a net with the established parameters
        :param X_train - Training data used to train the model
        :param y_train - Test data used to test the model
        :param layers - A list of values, where in each value denotes a layer, and the number of neurons for that layer
        :param loss_function - Function used to measure fitting of model (predicted from actual)
        :param optimizer - Function used to optimize the model (eg: Gradient Descent)
        """
        self.mode = mode
        self.model = ke.models.Sequential()
        #
        if len(layers) < 1:
            raise ValueError('Layer Count is Empty!')
        elif len(layers) == 1:
            self.model.add(ke.layers.LSTM(layers[0], input_shape=(X_train.shape[1], X_train.shape[2])))
        else:
            for i in range(0, len(layers) - 2):
                self.model.add(
                    ke.layers.LSTM(layers[i], input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
            self.model.add(ke.layers.LSTM(layers[i]))
        #
        # self.model.add(ke.layers.Dense(y_train.shape[1]))
        self.model.add(ke.layers.Dense(layers[-1]))
        self.model.add(ke.layers.Activation('sigmoid'))
        #
        if self.mode == 'regression':
            self.loss_func = 'mae'
        elif self.mode == 'classification':
            self.loss_func = 'categorical_crossentropy'
        else:
            self.loss_func = None
        #
        if self.mode == 'regression':
            self.model.compile(loss=self.loss_func, optimizer=optimizer)
        elif self.mode == 'classification':
            self.model.compile(loss=self.loss_func, optimizer=optimizer, metrics=['accuracy'])
        #
        # Map Discretize function to matrices
        self.vecfunc = np.vectorize(discretize_value)
        print(self.model.summary())

    #
    def fit_model(self, X_train, X_test, y_train, y_test, epochs=50, batch_size=50, verbose=2, shuffle=False,
                  plot=False):
        """
        Fit data to model & validate. Trains a number of epochs.
        """
        history = self.model.fit(x=X_train,
                                 y=y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=(X_test, y_test),
                                 verbose=verbose,
                                 shuffle=shuffle)
        if plot:
            plt.rcParams['figure.figsize'] = [20, 15]
            if self.mode == 'regression':
                plt.plot(history.history['loss'], label='train')
                plt.plot(history.history['val_loss'], label='validation')
            elif self.mode == 'classification':
                plt.plot(history.history['acc'], label='train')
                plt.plot(history.history['val_acc'], label='validation')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

    #
    def predict(self, X):
        yhat = self.model.predict(X)
        return yhat

    #
    def predict_and_evaluate(self, X, y, y_labels, plot=False):
        roc_values = []
        yhat = self.predict(X)
        #
        # RMSE Evaluation
        if self.mode == 'regression':
            rmse = math.sqrt(mean_squared_error(y, yhat))
            if not plot:
                return rmse
            print('Test ' + self.loss_func + ': %.3f\n-----------------------------\n\n' % rmse)
            #
            # F1-Score Evaluation
            for i in range(len(y_labels)):
                yv_c, yhat_c = [], []
                for val in y[:, i]:
                    yv_c.append(discretize_value(amount=val))
                for val in yhat[:, i]:
                    yhat_c.append(discretize_value(amount=val))
                f1 = f1_score(yv_c, yhat_c,
                              average='micro')  # Calculate metrics globally by counting the total true positives, false negatives and false positives.
                print('Test FScore [' + y_labels[i] + ']: ' + str(f1))
                #
                fpr_RF, tpr_RF, thresholds_RF = roc_curve(yv_c, yhat_c)
                auc_RF = roc_auc_score(yv_c, yhat_c)
                roc_values.append([fpr_RF, tpr_RF, thresholds_RF, auc_RF, y_labels[i]])
                print('AUC RF:%.3f' % auc_RF)
            #
            if plot:
                for fpr_RF, tpr_RF, thresholds_RF, auc_RF, label in roc_values:
                    plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC label[' + label + ']: %.3f' % auc_RF)
                plt.plot([0, 1], [0, 1], 'k-', label='random')
                plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
                plt.legend()
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.show()
        elif self.mode == 'classification':
            yhat = self.vecfunc(yhat)
            #
            # F1-Score Evaluation
            for i in range(len(y_labels)):
                f1 = f1_score(y[:, i], yhat[:, i],
                              average='micro')  # Calculate metrics globally by counting the total true positives, false negatives and false positives.
                print('Test FScore [' + y_labels[i] + ']: ' + str(f1))
                #
                fpr_RF, tpr_RF, thresholds_RF = roc_curve(y, yhat)
                auc_RF = roc_auc_score(y, yhat)
                roc_values.append([fpr_RF, tpr_RF, thresholds_RF, auc_RF, y_labels[i]])
                print('AUC RF:%.3f' % auc_RF)
            #
            if plot:
                for fpr_RF, tpr_RF, thresholds_RF, auc_RF, label in roc_values:
                    plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC label[' + label + ']: %.3f' % auc_RF)
                plt.plot([0, 1], [0, 1], 'k-', label='random')
                plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
                plt.legend()
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.show()
            else:
                return roc_values
        #
        if plot:
            for i in range(0, len(y[0])):
                plt.rcParams['figure.figsize'] = [20, 15]
                plt.plot(y[:, i], label='actual')
                plt.plot(yhat[:, i], label='predicted')
                plt.legend(['actual', 'predicted'], loc='upper left')
                plt.title(y_labels[i])
                plt.show()
    #
    @staticmethod
    def write_results_to_disk(path, iteration, lag, test_split, batch, depth, epoch, score, time_train):
        file_exists = os.path.isfile(path)
        with open(path, 'a') as csvfile:
            headers = ['iteration', 'lag', 'test_split', 'batch', 'depth', 'epoch', 'score', 'time_train']
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
            writer.writerow({'iteration': iteration,
                             'lag': lag,
                             'test_split': test_split,
                             'batch': batch,
                             'depth':depth,
                             'epoch':epoch,
                             'score': score,
                             'time_train': time_train})
#
# Test Harness
iteration = 0
max_lag_param = 15
test_harness_param = (.5,.6,.7,.8,.9)
batch=(10, 25, 50, 75, 100)
epochs=(100,125,150,175,200)
#
# Test Multiple Time Splits (Lag)
for lag in range(1,max_lag_param+1):
    t0 = time.time()
    shifted_df = series_to_supervised(df, lag, 1)
    #
    # Seperate labels from features
    y_df_column_names = shifted_df.columns[len(df.columns):len(df.columns) + len(y_label)]
    y_df = shifted_df[y_df_column_names]
    X_df = shifted_df.drop(columns=y_df_column_names)
    #
    # Delete middle timesteps
    X_df = remove_n_time_steps(data=X_df, n_in=lag)
    #
    # Test Multiple Train/Validation Splits
    for test_split in test_harness_param:
        X_train, X_validate, y_train, y_validate = train_test_split(X_df, y_df, test_size=test_split)
        X_train = X_train.values
        y_train = y_train.values
        X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=.5)
        X_validate = X_validate.values
        y_validate = y_validate.values
        #
        # Reshape for fitting in LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_validate = X_validate.reshape((X_validate.shape[0], 1, X_validate.shape[1]))
        hidden_net_depth = (int(X_train.shape[2]/2), int(X_train.shape[2]/1.5), int(X_train.shape[2]), int(X_train.shape[2]*1.5), int(X_train.shape[2]*2))
        #
        for epoch in epochs:
            for depth in hidden_net_depth:
                for batch_size in batch:
                    model = LSTM(X_train=X_train,
                                 y_train=y_train,
                                 layers=[X_train.shape[2],
                                         depth,
                                         len(y_label)],
                                 mode='regression')
                    model.fit_model(X_train=X_train,
                                    X_test=X_validate,
                                    y_train=y_train,
                                    y_test=y_validate,
                                    epochs=epoch,
                                    batch_size=batch_size,
                                    verbose=2,
                                    shuffle=False,
                                    plot=False)
                    rmse = model.predict_and_evaluate(X=X_validate,
                                                      y=y_validate,
                                                      y_labels=y_label,
                                                      plot=False)
                    t1 = time.time()
                    time_total = t1 - t0
                    LSTM.write_results_to_disk(path="time_series_lstm_regression_results.csv",
                                               iteration=iteration,
                                               lag=lag,
                                               test_split=test_split,
                                               batch=batch_size,
                                               depth=depth,
                                               epoch=epoch,
                                               score=rmse,
                                               time_train=time_total)
                    iteration += 1