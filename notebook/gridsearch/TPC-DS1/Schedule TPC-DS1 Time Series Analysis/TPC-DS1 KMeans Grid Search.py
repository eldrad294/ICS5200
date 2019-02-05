""" Module Import """

# scipy
import scipy as sc
print('scipy: %s' % sc.__version__)
# numpy
import numpy as np
print('numpy: %s' % np.__version__)
# matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
# pandas
import pandas as pd
from pandas.plotting import lag_plot
print('pandas: %s' % pd.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import sklearn as sk
print('sklearn: %s' % sk.__version__)
# Scipy
from scipy.spatial import distance
# Dimension Reduction Modules
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn.metrics import r2_score
# math
import math
import csv
import os.path
import time

""" Config Setup """

# Experiment Config
tpcds = 'TPCDS1'  # Schema upon which to operate test
lag = 13  # Time Series shift / Lag Step. Each lag value equates to 1 minute. Cannot be less than 1
if lag < 1:
    raise ValueError('Lag value must be greater than 1!')
test_split = .3  # Denotes which Data Split to operate under when it comes to training / validation
sub_sample_start = 350  # Denotes frist 0..n samples (Used for plotting purposes)
y_label = ['CPU_TIME_DELTA', 'IOWAIT_DELTA']  # Denotes which label to use for time series experiments
bin_value = 2
nrows = 100000
if bin_value < 2:
    raise ValueError('Number of buckets must be greater than 1')

iteration = 0
test_harness_param = (.2,.3,.4,.5)
# Forest Config
parallel_degree = 1
n_estimators = 300

# K-Means Config
n_clusters = bin_value
n_jobs = parallel_degree
initializers = ('k-means++', 'random')
random_state = None

# Root path
root_dir = 'C:/Users/gabriel.sammut/University/Data_ICS5200/Schedule/' + tpcds
#root_dir = 'D:/Projects/Datagenerated_ICS5200/Schedule/' + tpcds

# Open Data
rep_hist_snapshot_path = root_dir + '/rep_hist_snapshot.csv'
rep_hist_sysmetric_summary_path = root_dir + '/rep_hist_sysmetric_summary.csv'
rep_hist_sysstat_path = root_dir + '/rep_hist_sysstat.csv'
# rep_hist_snapshot_path = root_dir + '/rep_hist_snapshot.csv'
# rep_hist_sysmetric_summary_path = root_dir + '/rep_hist_sysmetric_summary.csv'
# rep_hist_sysstat_path = root_dir + '/rep_hist_sysstat.csv'

rep_hist_snapshot_df = pd.read_csv(rep_hist_snapshot_path, nrows=nrows)
rep_hist_sysmetric_summary_df = pd.read_csv(rep_hist_sysmetric_summary_path, nrows=nrows)
rep_hist_sysstat_df = pd.read_csv(rep_hist_sysstat_path, nrows=nrows)

def prettify_header(headers):
    """
    Cleans header list from unwated character strings
    """
    header_list = []
    [header_list.append(header.replace("(","").replace(")","").replace("'","").replace(",","")) for header in headers]
    return header_list

rep_hist_snapshot_df.columns = prettify_header(rep_hist_snapshot_df.columns.values)
rep_hist_sysmetric_summary_df.columns = prettify_header(rep_hist_sysmetric_summary_df.columns.values)
rep_hist_sysstat_df.columns = prettify_header(rep_hist_sysstat_df.columns.values)

print(rep_hist_snapshot_df.columns.values)
print(rep_hist_sysmetric_summary_df.columns.values)
print(rep_hist_sysstat_df.columns.values)

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

def get_na_columns(df, headers):
    """
    Return columns which consist of NAN values
    """
    na_list = []
    for head in headers:
        if df[head].isnull().values.any():
            na_list.append(head)
    return na_list

print('N/A Columns\n')
print('\n REP_HIST_SNAPSHOT Features ' + str(len(rep_hist_snapshot_df.columns)) + ': ' + str(get_na_columns(df=rep_hist_snapshot_df,headers=rep_hist_snapshot_df.columns)) + "\n")
print('REP_HIST_SYSMETRIC_SUMMARY Features ' + str(len(rep_hist_sysmetric_summary_df.columns)) + ': ' + str(get_na_columns(df=rep_hist_sysmetric_summary_df,headers=rep_hist_sysmetric_summary_df.columns)) + "\n")
print('REP_HIST_SYSSTAT Features ' + str(len(rep_hist_sysstat_df.columns)) + ': ' + str(get_na_columns(df=rep_hist_sysstat_df,headers=rep_hist_sysstat_df.columns)) + "\n")

def fill_na(df):
    """
    Replaces NA columns with 0s
    """
    return df.fillna(0)

# Populating NaN values with amount '0'
rep_hist_snapshot_df = fill_na(df=rep_hist_snapshot_df)
rep_hist_sysmetric_summary_df = fill_na(df=rep_hist_sysmetric_summary_df)
rep_hist_sysstat_df = fill_na(df=rep_hist_sysstat_df)

df = pd.merge(rep_hist_snapshot_df, rep_hist_sysmetric_summary_df,how='inner',on ='SNAP_ID')
df = pd.merge(df, rep_hist_sysstat_df,how='inner',on ='SNAP_ID')
print(df.shape)
print('----------------------------------')
print(df.columns.tolist())

df.sort_values(by=['SNAP_ID'], ascending=True, inplace=True)
print(df.shape)

df.astype('float32', inplace=True)
df = np.round(df, 3) # rounds to 3 dp
print(df.shape)


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

    # print('Features which are considered flatline:\n')
    # for col in flatline_features:
    #    print(col)
    print('\nShape before changes: [' + str(df.shape) + ']')
    df = df.drop(columns=flatline_features)
    print('Shape after changes: [' + str(df.shape) + ']')
    print('Dropped a total [' + str(len(flatline_features)) + ']')
    return df

print('Before column drop:')
print(df.shape)
df = drop_flatline_columns(df=df)
print('\nAfter flatline column drop:')
print(df.shape)
dropped_columns_df = ['PLAN_HASH_VALUE',
                      'OPTIMIZER_ENV_HASH_VALUE',
                      'LOADED_VERSIONS',
                      'VERSION_COUNT',
                      'PARSING_SCHEMA_ID',
                      'PARSING_USER_ID',
                      'CON_DBID',
                      'SNAP_LEVEL',
                      'SNAP_FLAG',
                      'COMMAND_TYPE']
df.drop(columns=dropped_columns_df, inplace=True)
print('\nAfter additional column drop:')
print(df.shape)


def get_outliers_quartile(df=None, headers=None):
    """
    Detect and return which rows are considered outliers within the dataset, determined by :quartile_limit (99%)
    """
    outlier_rows = []  # This list of lists consists of elements of the following notation [column,rowid]
    for header in headers:
        outlier_count = 0
        try:
            q25, q75 = np.percentile(df[header], 25), np.percentile(df[header], 75)
            iqr = q75 - q25
            cut_off = iqr * .6  # This values needs to remain as it. It was found to be a good value so as to capture the relavent outlier data
            lower, upper = q25 - cut_off, q75 + cut_off

            series_row = (df[df[header] > upper].index)
            outlier_count += len(list(np.array(series_row)))
            for id in list(np.array(series_row)):
                outlier_rows.append([header, id])

            series_row = (df[df[header] < lower].index)
            outlier_count += len(list(np.array(series_row)))
            for id in list(np.array(series_row)):
                outlier_rows.append([header, id])
            print(header + ' - [' + str(outlier_count) + '] outliers')
        except Exception as e:
            print(str(e))

    unique_outlier_rows = []
    for col, rowid in outlier_rows:
        unique_outlier_rows.append([col, rowid])
    return unique_outlier_rows

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

        for i in range(len(outliers)):
            if label == outliers[i][0]:
                df[label].iloc[outliers[i][1]] = mean_val + std_val
                # print('Header [' + str(outliers[i][0]) + '] - Location [' + str(outliers[i][1]) + '] - Value [' + str(df.iloc[outliers[i][1]][outliers[i][0]]) + ']')
    return df


print("DF with outliers: " + str(df.shape))
df = edit_outliers(df=df,
                   headers=y_label)
print("DF with edited outliers: " + str(df.shape))

class Normalizer:

    @staticmethod
    def robust_scaler(dataframe):
        """
        Normalize df using interquartile ranges as min-max, this way outliers do not play a heavy emphasis on the
        normalization of values.
        :param dataframe: (Pandas) Pandas data matrix
        :return: (Pandas) Normalized data matrix
        """
        headers = dataframe.columns
        X = preprocessing.robust_scale(dataframe.values)
        return pd.DataFrame(X, columns=headers)

    @staticmethod
    def minmax_scaler(dataframe):
        """
        Normalize df using min-max ranges for normalization method
        :param dataframe: (Pandas) Pandas data matrix
        :return: (Pandas) Normalized data matrix
        """
        headers = dataframe.columns
        X = preprocessing.minmax_scale(dataframe.values)
        return pd.DataFrame(X, columns=headers)

    @staticmethod
    def normalize(dataframe):
        """
        The normalizer scales each value by dividing each value by its magnitude in n-dimensional space for n number of features.
        :param dataframe: (Pandas) Pandas data matrix
        :return: (Pandas) Normalized data matrix
        """
        headers = dataframe.columns
        X = preprocessing.normalize(dataframe.values)
        return pd.DataFrame(X, columns=headers)

print('------------------BEFORE------------------')
print('------------------DF------------------')
print(df.shape)
print('\n')

# ROBUST SCALER
# df = Normalizer.robust_scaler(dataframe=df)

# MINMAX SCALER
df = Normalizer.minmax_scaler(dataframe=df)

# NORMALIZER
#df = Normalizer.normalize(dataframe=df)

print('\n\n------------------AFTER------------------')
print('------------------df------------------')
print(df.shape)
print('\n\n')
print('\n\ndf')
print(df.head())

y_label.append('SNAP_ID')
y_df = df[y_label]
del y_label[-1]
df.drop(columns=y_label, inplace=True)
print("Label " + str(y_label) + " shape: " + str(y_df.shape))
print("Feature matrix shape: " + str(df.shape))

# Merging labels and features in respective order
df = pd.merge(y_df,df,on='SNAP_ID',sort=False,left_on=None, right_on=None)
print('Merged Labels + Vectors: ' + str(df.shape))
print(df.head())


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
    if n_in != 0:
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    n_out += 1
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def remove_n_time_steps(data, n=1):
    """
    This method removes all timesteps up until the timestep denoted as the lag value.
    :param data: (Pandas dataframe) Denotes dataframe to have middle steps removed.
    :param n:    (Integer) Denotes lag value.
    :return: (Pandas dataframe) Denotes dataframe with removed itnermediate time lag values.
    """
    if n == 0:
        return data
    df = data
    headers = df.columns
    dropped_headers = []
    #     for header in headers:
    #         if "(t)" in header:
    #             dropped_headers.append(header)

    for i in range(1, n + 1):
        for header in headers:
            if "(t+" + str(i) + ")" in header:
                dropped_headers.append(str(header))

    return df.drop(dropped_headers, axis=1)


# Frame as supervised learning set
shifted_df = series_to_supervised(df, lag, lag)

# Seperate labels from features
y_row = []
for i in range(lag + 1, (lag * 2) + 2):
    y_df_column_names = shifted_df.columns[len(df.columns) * i:len(df.columns) * i + len(y_label)]
    y_row.append(y_df_column_names)
    print(y_df_column_names)
    print(type(y_df_column_names))
y_df_column_names = []
for row in y_row:
    for val in row:
        y_df_column_names.append(val)

# Seperate labels from features
# y_df_column_names = shifted_df.columns[len(df.columns)*lag:len(df.columns)*lag + len(y_label)]
y_df = shifted_df[y_df_column_names]
X_df = shifted_df.drop(columns=y_df_column_names)
print('\n-------------\nFeatures')
print(X_df.columns)
print(X_df.shape)
print('\n-------------\n:Labels')
print(y_df.columns)
print(y_df.shape)

class BinClass:
    """
    Takes data column, and scales them into discrete buckets. Parameter 'n' denotes number of buckets. This class needs
    to be defined before the LSTM class, since it is referenced during the prediction stage. Since Keras models output a
    continuous output (even when trained on discrete data), the 'BinClass' is required by the LSTM class.
    """

    @staticmethod
    def __validate(df, n):
        """
        Validates class parameters
        """
        if df is None:
            raise ValueError('Input data parameter is empty!')
        elif n < 2:
            raise ValueError('Number of buckets must be greater than 1')

    @staticmethod
    def __bucket_val(val, threshold, n):
        """
        Receives threshold value and buckets the val according to the passed threshold
        """
        for i in range(1, n+1):
            if val <= threshold * i:
                return i

    @staticmethod
    def discretize_value(X, n):
        """
        param: X - Input data
        param: n - Number of buckets
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 2)

        for i in range(X.shape[1]):
            max_val = X[:, i].max()
            threshold = max_val / n
            myfunc_vec = np.vectorize(lambda x: BinClass.__bucket_val(x, threshold, n))
            X[:, i] = myfunc_vec(X[:, i])
        return X


class FeatureEliminator:
    """
    This class is dedicated to housing logic pertaining to feature selection - retaining only labels which are considered
    important.
    """

    def __init__(self, X_df, y_df):
        """
        Class constructor.
        :param X_df: (Pandas) Pandas feature matrix.
        :param y_df: (Pandas) Pandas label matrix.
        :return: None
        """
        self.__X_df = X_df
        self.__y_df = y_df

    def rfe_selector(self, test_split=.4, optimum_feature_count=0, parallel_degree=1, max_depth=None,
                     max_features='sqrt', n_estimators=100):
        """
        Recursive Feature Elimination Function. Isolates and eliminated features one by one, up till the desired amount, starting
        by features which are considered less important.
        :param test_split:            (Float) Denotes training/testing data split.
        :param optimum_feature_count: (Integer) Denotes the best estimated number of features to retain before a performance drop
                                                is estimated.
        :param parallel_degree:       (Integer) Denotes model training parallel degree.
        :param max_depth:             (Integer) Denotes number of leaves to evaluate during decision tree pruning.
        :param max_features:          (Integer) Denotes number of features to consider during random subselection.
        :param n_estimators:          (Integer) Number of estimators (trees) to build for decision making.
        :return: (List) This list is composed of boolean values, which correspond to the input feature column headers. True List
                        values denote columns which have been retained. False values denote eliminated feature headers.
        :return: (List) This list denotes feature rankings, which correspond to the input feature column headers. Values of '1',
                        denote that features have been retained.
        """
        X_df = self.__X_df.values
        y_df = self.__y_df[self.__y_df.columns[
            0]].values  # We can only use a single target column since RandomForests do not support multi target labels
        print(X_df.shape)
        print(y_df.shape)
        optimum_feature_count = int(optimum_feature_count)

        X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                            y_df,
                                                            test_size=test_split)
        model = RandomForestRegressor(n_estimators=int(n_estimators),
                                      n_jobs=parallel_degree,
                                      max_depth=max_depth,
                                      max_features='sqrt')

        # create the RFE model and select N attributes
        rfe_model = RFE(model, optimum_feature_count, step=1)
        rfe_model = rfe_model.fit(X_train, y_train)

        # summarize the selection of the attributes
        print(rfe_model.support_)
        print(rfe_model.ranking_)

        # evaluate the model on testing set
        pred_y = rfe_model.predict(X_test)
        predictions = [round(value) for value in pred_y]
        r2s = r2_score(y_test, predictions)

        return rfe_model.support_, rfe_model.ranking_

    def get_selected_features(self, column_mask):
        """
        Retrieves features which have not been eliminated from the RFE function.
        :param column_mask: (List) This list is composed of boolean values, which correspond to the input feature column headers.
                                   True list values denote columns which have been retained. False values denote eliminated
                                   feature headers.
        :return: (Pandas) Pandas data matrix.
        """
        recommended_columns = []
        for i in range(len(self.__X_df.columns)):
            if (column_mask[i]):
                recommended_columns.append(self.__X_df.columns[i])

        return self.__X_df[recommended_columns]

fe = FeatureEliminator(X_df=X_df,
                       y_df=y_df)
column_mask, column_rankings = fe.rfe_selector(test_split=test_split,
                                               optimum_feature_count=X_df.shape[1]/4,
                                               parallel_degree=parallel_degree,
                                               max_depth=1,
                                               max_features='sqrt',
                                               n_estimators=n_estimators)
print(X_df.columns)
X_df = fe.get_selected_features(column_mask=column_mask)
print(X_df.columns)

class PrincipalComponentAnalysisClass:
    """
    This class handles logic related to PCA data transformations.
    https://towardsdatascience.com/an-approach-to-choosing-the-number-of-components-in-a-principal-component-analysis-pca-3b9f3d6e73fe
    """

    def __init__(self, X_df):
        """
        Cosntructor method.
        :param X_df: (Pandas) Dataframe consisting of input features, which will be subject to PCA.
        :return: None
        """
        self.__X_df = X_df

    def get_default_component_variances(self):
        """
        Fitting the PCA algorithm with our Data.
        :return: (Numpy array) Array of feature variances.
        """
        pca = PCA().fit(self.__X_df.values)
        return np.cumsum(pca.explained_variance_ratio_)

    def get_default_component_count(self, threshold=.99):
        """
        Retrieves the recommended number of component decomposition, above which very little variance
        gain is achieved. This treshold will be set at a 0.999 variance threshold.
        :param threshold: (Float) Threshold value between 0 and 1. Stops immediately as soon the number
                                  of required components exceeds the threshold value.
        :return: (Integer) Returns the number of recommended components.
        """
        variance_ratios = self.get_default_component_variances()
        n = 0
        for val in variance_ratios:
            if val < threshold:
                n += 1
        return n

    def plot_variance_per_reduction(self):
        """
        This method subjects the feature matrix to a PCA decomposition. The number of components is plot
        vs the amount of retained variance.
        :return: None
        """
        variance_ratios = self.get_default_component_variances()

        # Plotting the Cumulative Summation of the Explained Variance
        plt.figure()
        plt.plot(variance_ratios)
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)')  # for each component
        plt.title(tpcds + ' Dataset Explained Variance')
        plt.show()

    def apply_PCA(self, n_components):
        """
        Applies Principle Component Analysis on the constructor passed data matrix, on a number of components.
        A new pandas data matrix is returned, with renamed 'Principal Component' headers.
        :param n_components: (Integer) Denotes number of component breakdown.
        :return: (Pandas) Dataframe consisting of new decomposed components.
        """
        pca = PCA(n_components=n_components)
        dataset = pca.fit_transform(self.__X_df.values)
        header_list = []
        for i in range(dataset.shape[1]):
            header_list.append('Component_' + str(i))
        return pd.DataFrame(data=dataset, columns=header_list)


print(X_df.head())
print(X_df.shape)

pcac = PrincipalComponentAnalysisClass(X_df=X_df)
# pcac.plot_variance_per_reduction()
component_count = pcac.get_default_component_count()
X_df = pcac.apply_PCA(n_components=component_count)

print('-' * 30)
print(X_df.head())
print(X_df.shape)


class KMeansWrapper:
    """
    A class which serves a K-Means wrapper
    """

    def __init__(self, n_clusters, n_jobs, init, random_state, n_buckets, y_label, lag):
        """
        Creates a K-Means back, a list which holds a number of K-Means models (each of which will be trained
        according to label). Number of K Models equates to number of (labels * number of categories)
        :param: n_clusters - Hyper parameter 'K'
        :param: n_jobs - Model parallel degree
        :param: init - Initialization mode hyper parameter
        :param: random_state - Random seed
        :param: n_buckets - Number of categories
        :param: y_label - List columns consisting of labels
        :param: lag - Integer denoting lag value
        :return: None
        """
        self.__K_bank = {}
        self.y_label = y_label
        self.lag = lag
        for i in range(len(self.y_label) * self.lag):
            k_dict = {}
            for j in range(1, n_buckets + 1):
                k_dict[j] = KMeans(n_clusters=n_clusters,
                                   n_jobs=n_jobs,
                                   init=init,
                                   random_state=random_state)
            self.__K_bank[y_label[i % len(y_label)]] = k_dict

        self.n_buckets = n_buckets

    def fit_model_bank(self, X, y):
        """
        Fits data to K-Means models, according to label
        :param: X - Training data matrix
        :param: y - Training labels
        :return: None
        """

        y_row_count = y.shape[0]

        # For every label in the validation/text set
        for i in range(len(self.y_label) * self.lag):

            # For every class category
            X_dict = {}
            for j in range(1, self.n_buckets + 1):
                X_dict[j] = []

            # print('Training ' + str(self.y_label[i % len(self.y_label)]) + ' model with lag ' + str(
            #     math.ceil((i + 1) / len(self.y_label))) + '..')
            for col_idx in range(0, len(y[:, i])):
                for j in range(1, self.n_buckets + 1):
                    if j == y[col_idx, i]:
                        X_dict[j].append(X[col_idx, :])
                        break

            for j in range(1, self.n_buckets + 1):
                if len(X_dict[j]) != 0:

                    X_dict[j] = np.asarray(X_dict[j])

                    try:
                        self.__K_bank[str(self.y_label[i % len(self.y_label)])][j].fit(X_dict[j])
                    except ValueError:

                        # This exception handler is required for training cases when the number of label
                        # categories is less than the number of preset clusters. In such cases, 'K' is
                        # modified to match the number of label categories.

                        if X_dict[j].shape[0] < bin_value:
                            # Fitting model with a 'K' equivalent to number of label categories
                            self.__K_bank[str(y_label[i % len(self.y_label)])][j] = KMeans(
                                n_clusters=X_dict[j].shape[0],
                                n_jobs=n_jobs,
                                init=init,
                                random_state=random_state)

                            self.__K_bank[str(y_label[i % len(self.y_label)])][j].fit(X_dict[j])

    def predict(self, X):
        """
        Iterates over all labels, and finds closest average centroid cluster. Closest cluster (euclidean) is
        determined as the target label.
        :param: X - Training data matrix
        :return: Numpy array (Predicted Labels)
        """
        yhat_tot = []
        for row_i in range(X.shape[0]):
            yhat = []
            for i in range(len(y_label) * self.lag):
                k_models_dict = self.__K_bank[str(self.y_label[i % len(self.y_label)])]
                avg_centroids = []
                for key, val in k_models_dict.items():
                    try:
                        avg_centroid = self.__calc_avg_centroid(val.cluster_centers_)
                        avg_centroids.append(avg_centroid)
                    except AttributeError:
                        # This exception handle is for scenarios where a K-Means model has not been initiated
                        # due to not having respective labels in the training dataset. In such scenarious,
                        # cluster_centers_ will raise an attribute error
                        continue

                # Calculate euclidean distance between input vector and centroids
                distances = []
                for avg_c in avg_centroids:
                    distances.append(distance.euclidean(avg_c, X[row_i, :]))

                    # Compute shortest distance (Find nearest centroid cluster)
                for i in range(len(distances)):
                    if distances[i] == min(distances):
                        yhat.append(i)
                        break
            yhat_tot.append(yhat)
        return np.asarray(yhat_tot)

    def evaluate(self, y, yhat, plot=False):
        """
        Iterates over all validate/test data and calculated metrics per label.
        :param: y    - Numpy 2D array consisting of training data matrix
        :param: yhat - Numpy 2D array consisting of training labels
        :param: plot - Boolean value denoting whether this function should plot out it's evaluation
        :return: None
        """
        y = y.flatten()
        yhat = yhat.flatten()
        accuracy = accuracy_score(y, yhat)
        f1 = f1_score(y, yhat, average='macro')

        if plot:
            plt.rcParams['figure.figsize'] = [20, 15]
            plt.plot(y, label='actual')
            plt.plot(yhat, label='predicted')
            plt.legend(['actual', 'predicted'], loc='upper left')
            plt.title('Actual vs Predicted')
            plt.show()
        else:
            return accuracy, f1

    def get_labels(self):
        """
        Returns labels derived from cluster centroids.
        :return: (List) Model labels in the form of a list.
        """
        return self.model.labels_

    def get_K_bank(self):
        """
        Retrieves dictionary of trained K-Models.
        :return: (Dictionary) of K-Means models
        """
        return self.__K_bank

    def __calc_avg_centroid(self, centroids):
        """
        Calculates a middle/average vector based on the input centroids
        :param: centroids - Feature vectors (Numpy Array) concerning centroid cooridinates
        :return: (Numpy Array) Average vector of all input centroids
        """
        return np.average(centroids)

    @staticmethod
    def write_results_to_disk(path, iteration, lag, test_split, K, init, random_run,  rmse, accuracy, f_score,
                              time_train):
        """
        Static method which is used for test harness utilities. This method attempts a grid search across many
        trained K-Means models, each denoted with different configurations.
        Attempted configurations:
        * Varied lag projection.
        * Varied data test split.
        * Varied K parameter values.
        Each configuration is denoted with a score, and used to identify the most optimal configuration.

        :param: path         - String denoting path towards result csv output
        :param: iteration    - Integer denoting test iteration (Unique per test configuration)
        :param: lag          - Integer denoting lag value
        :param: test_split   - Float denoting data sample sizes
        :param: K            - (Integer) Clustering hyper parameter.
        :param: init         - (String) Denotes K-Means centroid starting position.
        :param: random_run   - (Integer) Denotes iteration of K-means run (every K-Means test is executed 5 times to
                                cater for random centroid initializing positions)
        :param: rmse         - (Float) Float denoting experiment configuration RSME score.
        :param: accuracy     - (Float) Float denoting experiment accuracy score.
        :param: fscore       - (Float) Float denoting experiment fscore score.
        :param: time_train   - Integer denoting number of seconds taken by LSTM training iteration
        :return: None
        """
        file_exists = os.path.isfile(path)
        with open(path, 'a') as csvfile:
            headers = ['iteration', 'lag', 'test_split', 'K', 'rmse', 'init', 'random_run', 'accuracy', 'f_score', 'time_train']
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
            writer.writerow({'iteration': iteration,
                             'lag': lag,
                             'test_split': test_split,
                             'K': K,
                             'init':init,
                             'random_run':random_run,
                             'rmse': rmse,
                             'accuracy': accuracy,
                             'f_score': f_score,
                             'time_train': time_train})

X_df = pd.DataFrame(BinClass.discretize_value(X_df.values, bin_value), columns=X_df.columns)
y_df = pd.DataFrame(BinClass.discretize_value(y_df.values, bin_value), columns=y_df.columns)
print(np.unique(X_df.values))
print(np.unique(y_df.values))

X_df = X_df.apply(LabelEncoder().fit_transform)
y_df = y_df.apply(LabelEncoder().fit_transform)
print(np.unique(X_df.values))
print(np.unique(y_df.values))

for test_split in test_harness_param:
    X_train, X_validate, y_train, y_validate = train_test_split(X_df, y_df, test_size=test_split)
    X_train = X_train.values
    y_train = y_train.values
    X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=.5)
    X_validate = X_validate.values
    y_validate = y_validate.values

    # Start at K=4 since K=2 was attempted in cell above.
    for init in initializers:
        for random_run in range(0, 5):
            for k in range(2, X_train.shape[0]):
                t0 = time.time()
                print('\n\nK parameter = ' + str(k))
                model = KMeansWrapper(n_clusters=k,
                                      n_jobs=n_jobs,
                                      init=init,
                                      random_state=random_state,
                                      n_buckets=bin_value,
                                      y_label=y_label,
                                      lag=lag)

                # Train on discrete data (Train > Validation)
                print('-----------------Training / Validation-----------------')
                model.fit_model_bank(X=X_train,
                                     y=y_train)

                acc_list, f_list = [], []
                for i in range(0, X_validate.shape[0]):
                    X = np.array([X_validate[i, :]])
                    y = model.predict(X)
                    if y.shape[1] == 0:
                        continue
                    model.fit_model_bank(X=X,
                                         y=y)

                    acc_score, f_score = model.evaluate(y=y_validate[i, :],
                                                        yhat=y,
                                                        plot=False)
                    acc_list.append(acc_score)
                    f_list.append(f_score)

                t1 = time.time()
                time_total = t1 - t0
                if len(acc_list) > 0:
                    KMeansWrapper.write_results_to_disk(path='time_series_clustering_classification_results.csv',
                                                        iteration=iteration,
                                                        lag=lag,
                                                        test_split=test_split,
                                                        K=k,
                                                        init=init,
                                                        random_run=random_run,
                                                        rmse=None,
                                                        accuracy=sum(acc_list) / len(acc_list),
                                                        f_score=sum(f_list) / len(f_list),
                                                        time_train=time_total)

                    print('----------------------------' + str(iteration) + '----------------------------')
                    iteration += 1