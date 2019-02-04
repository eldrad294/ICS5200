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
# pandas
import pandas as pd
print('pandas: %s' % pd.__version__)
# scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.ensemble import IsolationForest
import sklearn as sk
print('sklearn: %s' % sk.__version__)
# theano
import theano
print('theano: %s' % theano.__version__)
# tensorflow
import tensorflow
print('tensorflow: %s' % tensorflow.__version__)
# plaidml keras
#import plaidml.keras
#plaidml.keras.install_backend()
# keras
import keras as ke
print('keras: %s' % ke.__version__)
import math, csv, os, time

""" Configuration Cell """

# Experiment Config
tpcds='TPCDS1' # Schema upon which to operate test
lag=13 # Time Series shift / Lag Step. Each lag value equates to 1 minute. Cannot be less than 1
if lag < 1:
    raise ValueError('Lag value must be greater than 1!')

nrows=None
iteration=0
test_harness_param = (.2, .3, .4, .5) # Denotes which Data Split to operate under when it comes to training / validation

# Top Consumer Identification
y_label = ['COST','CARDINALITY','BYTES','IO_COST','TEMP_SPACE','TIME']
black_list = ['TIMESTAMP','SQL_ID'] # Columns which will be ignored during type conversion, and later used for aggregation
contamination = .1
parallel_degree = 1

# Net Config
max_epochs = (50, 100, 150)
max_batch = (32, 64, 128)
layers = (1, 2, 3)
drop_out = (0, .2, .4)
activations = ('selu', 'tanh', 'sigmoid')
initializers = ('glorot_normal', 'uniform', 'normal')
state=False

# Root path
#root_dir = 'C:/Users/gabriel.sammut/University/Data_ICS5200/Schedule/' + tpcds
root_dir = 'D:/Projects/Datagenerated_ICS5200/Schedule/' + tpcds

# Open Data
#rep_hist_snapshot_path = root_dir + '/rep_hist_snapshot.csv'
rep_hist_snapshot_path = root_dir + '/rep_hist_snapshot.csv'
rep_vsql_plan_path = root_dir + '/rep_vsql_plan.csv'

rep_hist_snapshot_df = pd.read_csv(rep_hist_snapshot_path, nrows=nrows)
rep_vsql_plan_df = pd.read_csv(rep_vsql_plan_path, nrows=4000000)

def prettify_header(headers):
    """
    Cleans header list from unwated character strings
    """
    header_list = []
    [header_list.append(header.replace("(","").replace(")","").replace("'","").replace(",","")) for header in headers]
    return header_list

rep_hist_snapshot_df.columns = prettify_header(rep_hist_snapshot_df.columns.values)
rep_vsql_plan_df.columns = prettify_header(rep_vsql_plan_df.columns.values)

print(rep_hist_snapshot_df.columns.values)
print('------------------------------------------')
print(rep_vsql_plan_df.columns)

""" Dealing with empty values """

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
print('REP_VSQL_PLAN Features ' + str(len(rep_vsql_plan_df.columns)) + ': ' + str(get_na_columns(df=rep_vsql_plan_df,headers=rep_vsql_plan_df.columns)) + "\n")

def fill_na(df):
    """
    Replaces NA columns with 0s
    """
    df = df.replace('', 0)
    return df.fillna(0)

# Populating NaN values with amount '0'
rep_hist_snapshot_df = fill_na(df=rep_hist_snapshot_df)
rep_vsql_plan_df = fill_na(df=rep_vsql_plan_df)

""" Type Conversion """


def handle_numeric_overflows(x):
    """
    Accepts a dataframe column, and
    """
    try:
        # df = df.astype('int64')
        x1 = pd.DataFrame([x], dtype='int64')
    except ValueError:
        x = 9223372036854775807  # Max int size
    return x


for col in rep_vsql_plan_df.columns:
    try:
        rep_vsql_plan_df[col] = rep_vsql_plan_df[col].astype('int64')
    except OverflowError:

        # Handles numeric overflow conversions by replacing such values with max value inside the dataset.
        rep_vsql_plan_df[col] = rep_vsql_plan_df[col].apply(handle_numeric_overflows)
        rep_vsql_plan_df[col] = rep_vsql_plan_df[col].astype('int64')
    except Exception as e:
        if col not in black_list:
            rep_vsql_plan_df.drop(columns=col, inplace=True)
            print('Dropped column [' + col + ']')

print(rep_hist_snapshot_df.columns)
print(rep_vsql_plan_df.columns)

""" Changing Matrix Shapes """

print("Shape Before Aggregation: " + str(rep_hist_snapshot_df.shape))
#
# Group By Values by SNAP_ID , sum all metrics (for table REP_HIST_SNAPSHOT) and drop all numeric
df = rep_hist_snapshot_df.groupby(['SNAP_ID'])['SQL_ID'].apply(list).reset_index()
#
print("Shape After Aggregation: " + str(df.shape))
print(type(df))
print(df.head(100))

""" Data Ordering """

df.sort_index(ascending=True,inplace=True)
print(df.shape)
print(df.head(100))

""" Univariate Selection """

print(df.shape)
del df['SNAP_ID']
print(df.shape)

""" Top SQL Identification """


class IsolationForestWrapper:
    """
    This class wraps up logic to the Isolation Forest Outlier Detection functionality.
    """

    def __init__(self, X, contamination=.1, parallel_degree=1):
        """
        Constructor Method

        :param X - Pandas Dataframe
        :param contamination - Real value
        :param parallel_degree - Parellization parameter

        :return: None
        """
        self.X = X.values
        self.model = IsolationForest(n_estimators=100, max_samples=256, contamination=contamination, random_state=0,
                                     n_jobs=parallel_degree)
        self.model.fit(self.X)
        self.scorings = []
        print(self.model)

    def __get_threshold_vector(self):
        """
        Calculates a vector threshold, above which will be used to identify outliers. This method is used for evaluating the
        trained machine-learning model.

        :return: Numpy vector which represents a threshold vector
        """
        mean = np.mean(self.X)
        std = np.std(self.X)
        std3 = np.multiply(std, 3)
        return np.add(mean, std3)

    def __calculate_expected_labels(self):
        """
        Estimates label clustering by comparing them to a threshold mean value. These labels will be used to gauge a scoring
        for the unsupervised clustering achieved by the IForest algorithm.

        :return: A list of the expected output labels.
        """
        mean_vect = self.__get_threshold_vector()
        mean_labels = []
        for vector in self.X:
            if np.greater(vector, mean_vect).any():
                mean_labels.append(-1)
            else:
                mean_labels.append(1)
        return mean_labels

    def retrieve_scorings(self):
        """
        This method retrieves the per vector IForest scorings, after the model has been trained.

        :return: List of Iforest scorings
        """
        return self.model.decision_function(self.X)

    def plot_scorings(self):
        """
        Distributes into 50 bin histogram.

        :return: None
        """
        scores = self.retrieve_scorings()
        plt.figure(figsize=(12, 8))
        plt.hist(scores, bins=50);
        plt.title('Isolation Forest Scorings')
        plt.show()

    def predict_labels(self):
        """
        Caries out predicton on feature matrix 'X'

        :return: List of predicted output labels.
        """
        return self.model.predict(self.X)

    def predict_labels(self, X):
        """
        Caries out predicton on feature matrix 'X'

        :return: List of predicted output labels.
        """
        return self.model.predict(X)

    def outlier_score_accuracy(self):
        """
        Returns a score which evaluates the accuracy with the number of isolated outliers. The closer to 0 the score, the more accurate the evaluation

        :return: Positive Integer (Squared and Square Rooted) denoting the delta scoring between predicted and actual
        """
        if self.scorings is None or len(self.scorings) == 0:
            raise ValueError('Scorings list is empty!')
        elif len(self.scorings) > 2:
            raise ValueError(
                'Scorings list length is greater than 2! Must be composed of the following structure [scoring1, scoring2]')

        return math.sqrt((self.scorings[1] - self.scorings[0]) ** 2)

    def evaluate_labels(self):
        """
        This function calculates the expected inlier and outlier vectors based on a statistical threshold, and then matches
        these expectations to the IForest predictions. Results are plotted, and gauge by scored ROC score, and lowest error delta

        :return: None
        """
        y = self.__calculate_expected_labels()
        yhat = self.predict_labels()

        unique, counts = np.unique(y, return_counts=True)
        print('Expected Label Distribution')
        for i in range(len(unique)):
            print('Label [' + str(unique[i]) + '] -> Count [' + str(counts[i]) + ']')
            if unique[i] == -1:
                self.scorings.append(counts[i])
        unique, counts = np.unique(yhat, return_counts=True)
        print('Isolated Label Distribution')
        for i in range(len(unique)):
            print('Label [' + str(unique[i]) + '] -> Count [' + str(counts[i]) + ']')
            if unique[i] == -1:
                self.scorings.append(counts[i])

        print("\n----\nAccuracy: " + str(accuracy_score(y, yhat)))
        print("F-Score: " + str(f1_score(y, yhat, average='binary')))
        print('---')
        print("Outlier Score Precision [" + str(self.outlier_score_accuracy()) + "]")

        fpr_RF, tpr_RF, thresholds_RF = roc_curve(y, yhat)
        print(fpr_RF)
        print(tpr_RF)
        auc_RF = roc_auc_score(y, yhat)
        print('AUC RF:%.3f' % auc_RF)
        plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
        plt.plot([0, 1], [0, 1], 'k-', label='random')
        plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
        plt.legend()
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()


ifw = IsolationForestWrapper(X=rep_vsql_plan_df[y_label], contamination=contamination, parallel_degree=parallel_degree)
# ifw.plot_scorings()

""" Stripping Inlier SQL_ID's """

sql_ids = np.unique(rep_vsql_plan_df['SQL_ID'].values)
sql_map = {}
for sql in sql_ids:
    df_plan = rep_vsql_plan_df.loc[rep_vsql_plan_df['SQL_ID'] == sql]
    plan_costings = df_plan[y_label]
    sql_map[sql] = ifw.predict_labels(plan_costings.values)[0]
print(sql_map)

outlier_ids = []
for key, value in sql_map.items():
    if value == -1:  # -1 Denotes Outliers, as predicted by the isolation forst
        outlier_ids.append(key)
print('Outlier SQL_IDs: \n' + str(outlier_ids))

for index, row_sql_ids in df.iterrows():
    snap_list = []
    for sql_id in row_sql_ids['SQL_ID']:
        if sql_id in outlier_ids:
            snap_list.append(sql_id)
    df['SQL_ID'].iloc[index] = snap_list
print(df)

""" Label Encoding """


class LabelEncoder:
    """
    Scikit Label Encoder was acting up with the following error whilst using the transform function, even though I tripled
    checked that the passed data was exactly the same as the one used for training:

    * https://stackoverflow.com/questions/46288517/getting-valueerror-y-contains-new-labels-when-using-scikit-learns-labelencoder

    So I have rebuilt a similar functionality to categorize my data into numeric digits, as the LabelEncoder is supposed to do.
    """

    def __init__(self):
        self.__class_map = {}
        self.__integer_counter = 0

    def fit(self, X):
        """
        :param - X: python list
        """
        for val in X:
            if val not in self.__class_map:
                self.__class_map[val] = self.__integer_counter
                self.__integer_counter += 1

    def transform(self, X):
        """
        param - X: python list
        """
        encoded_map = []
        for val in X:
            if val in self.__class_map:
                value = self.__class_map[val]
                encoded_map.append(value)
            else:
                raise ValueError('Label Mismatch - Encountered a label which was not trained on.')
        return encoded_map

    def get_class_map(self):
        """
        Returns original classes as a list
        """
        class_map = []
        for key, value in self.__class_map.items():
            class_map.append(key)
        return class_map

    def get_encoded_map(self):
        """
        Returns class encodings as a list
        """
        encoded_map = []
        for key, value in self.__class_map.items():
            encoded_map.append(value)
        return encoded_map
    
print(df.shape)
print(df.head(10))
le = LabelEncoder()
for index, row in df.iterrows():
    sql_id_list = row['SQL_ID']
    le.fit(sql_id_list)
for index, row in df.iterrows():
    sql_id_list = row['SQL_ID']
    transformed_list = le.transform(sql_id_list)
    df['SQL_ID'].iloc[index] = transformed_list 

print("\n----------------------------------\n\nAvailable Classes:")
print('Total SQL_ID Classes: ' + str(len(le.get_class_map())))
print(le.get_class_map()[:10])
print(df.shape)
print(df.head(10))

""" Feature Padding """

print("Length at index 0: " + str(len(df['SQL_ID'].iloc[0])))
print(df['SQL_ID'].iloc[0])
print("Length at index 1: " + str(len(df['SQL_ID'].iloc[1])))
print(df['SQL_ID'].iloc[1])
print("Length at index 2: " + str(len(df['SQL_ID'].iloc[2])))
print(df['SQL_ID'].iloc[2])


# Retrieve largest length
def pad_datamatrix(df):
    """
    Iterates over dataframe and pads SQL_ID lists accordingly with -1 values, denoting empty SQL_ID slots.
    """
    row_sizes = []
    for index, row in df.iterrows():
        row_sizes.append(len(row['SQL_ID']))
    max_row_size = max(row_sizes)

    # Pad Dataframe Values
    i = 0
    for index, row in df.iterrows():
        length = len(row['SQL_ID'])
        diff = max_row_size - length
        if diff != 0:
            for j in range(length, max_row_size):
                df['SQL_ID'].iloc[i] = np.append(df['SQL_ID'].iloc[i], -1)  # Appends -1 to padded values
        # print("Length at index " + str(i) + ": " + str(df['SQL_ID'].iloc[i].size))
        i += 1
    return df


df = pad_datamatrix(df)

print('\n\n------------------------------------------\n\n')
print("Length at index 0: " + str(len(df['SQL_ID'].iloc[0])))
print(df['SQL_ID'].iloc[0])
print("Length at index 1: " + str(len(df['SQL_ID'].iloc[1])))
print(df['SQL_ID'].iloc[1])
print("Length at index 2: " + str(len(df['SQL_ID'].iloc[2])))
print(df['SQL_ID'].iloc[2])

""" Expand Feature Lists """


def sequence2features(df):
    """
    Converts pandas sequences into full fledged columns/features
    """
    feature_count = len(df[df.columns[0]].iloc[0])
    for column_name in df.columns:
        data_matrix = []
        new_values = df[column_name].values

        new_values = np.stack(new_values, axis=0)

        for i in range(1, feature_count + 1):
            new_column_name = column_name + "_" + str(i)
            df[new_column_name] = new_values[:, i - 1]

        # Drop original list columns
        df.drop(column_name, inplace=True, axis=1)
    return df


print('Features')
print('Before: ' + str(df.shape))
df = sequence2features(df=df)
print('After: ' + str(df.shape))

""" One Hot Encoder """


class OneHotEncoder:

    def __init__(self, classes):
        self.__mapper = pd.DataFrame(columns=classes)

    def fit_transform(self, X):
        class_types = self.__mapper.columns
        for row in X:
            temp_row = []
            for i in range(len(class_types)):
                if class_types[i] in row:
                    temp_row.append(float(1))
                else:
                    temp_row.append(float(0))
            self.__mapper.loc[len(self.__mapper)] = temp_row
        return self.__mapper

    def get_classes(self):
        return self.__mapper.columns

    def get_unique_values(self):
        return np.unique(self.__mapper.values)

# One Hot Encoding train data
ohe = OneHotEncoder(classes=le.get_encoded_map())
print('Training Data:')
print("Before One Hot Encoding: " + str(df.shape))
df = ohe.fit_transform(X=df.values)
print("After One Hot Encoding: " + str(df.shape))
print(df)
print('Value type: ' + str(ohe.get_unique_values()))
print(type(df))

""" Time Series Shifting """

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
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    n_out += 1
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

def remove_n_time_steps(data, n=1):
    if n == 0:
        return data
    df = data
    headers = df.columns
    dropped_headers = []
    #
    for i in range(1,n+1):
        for header in headers:
            if "(t+"+str(i)+")" in header:
                dropped_headers.append(str(header))
    #
    return df.drop(dropped_headers, axis=1)

# Frame as supervised learning set
shifted_df = series_to_supervised(df, lag, lag)

# Seperate labels from features
x_columns, y_columns = [], []
for col in shifted_df.columns:
    if '+' in col:
        y_columns.append(col)
    else:
        x_columns.append(col)

y_df = shifted_df[y_columns]
X_df = shifted_df[x_columns]
print('\n-------------\nFeatures')
print(X_df.columns)
print(X_df.shape)
print(type(X_df))
print('\n-------------\nLabels')
print(y_df.columns)
print(y_df.shape)
print(type(y_df))

# # Delete middle timesteps
# X_df = remove_n_time_steps(data=X_df, n=lag)
# print('\n-------------\nFeatures After Time Shift')
# print(X_df.columns)
# print(X_df.shape)
# print(type(X_df))
# # y_df = remove_n_time_steps(data=y_df, n=lag)
# print('\n-------------\nLabels After Time Shift')
# print(y_df.columns)
# print(y_df.shape)
# print(type(y_df))

""" Discrete Training """

class BinClass:
    """
    Takes data column, and scales them into discrete buckets. Parameter 'n' denotes number of buckets. This class needs
    to be defined before the NeuralNet class, since it is referenced during the prediction stage. Since Keras models output a
    continuous output (even when trained on discrete data), the 'BinClass' is required by the NeuralNet class.
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
                return i-1

    @staticmethod
    def discretize_value(X, n):
        """
        param: X - Input data
        param: n - Number of buckets
        """
        if len(X.shape) == 1:
            X = np.reshape(X, (-1, 1))

        for i in range(X.shape[1]):
            max_val = X[:,i].max()
            threshold = max_val / n
            myfunc_vec = np.vectorize(lambda x: BinClass.__bucket_val(x, threshold, n))
            X[:,i] = myfunc_vec(X[:,i])
        return X

""" Deep Learning Model (LSTM) """

# LSTM Class
class LSTM:
    """
    Long Short Term Memory Neural Net Class
    """

    def __init__(self, X, y, lag, loss_func, activation, optimizer='sgd', lstm_layers=1, dropout=.0,
                 stateful=False, initializer='uniform'):
        """
        Initiating the class creates a net with the established parameters
        :param X             - (Numpy 2D Array) Training data used to train the model (Features).
        :param y             - (Numpy 2D Array) Test data used to test the model (Labels
        :param lag           - (Integer) Denotes lag step value
        :param loss_function - (String)  Denotes mode of measure fitting of model (Fitting function).
        :param activation    - (String)  Neuron activation function used to activate/trigger neurons.
        :param optimizer     - (String)  Denotes which function to us to optimize the model build (eg: Gradient Descent).
        :param lstm_layers   - (Integer) Denotes the number of LSTM layers to be included in the model build.
        :param dropout       - (Float)   Denotes amount of dropout for model. This parameter must be a value between 0 and 1.
        :param stateful      - (Boolean) Denotes whether state is used as initial state for next training batch.
        :param: initializer  - (String)  String initializer which denotes starting weights.
        """
        self.__lag = lag
        self.__model = ke.models.Sequential()

        if dropout > 1 and dropout < 0:
            raise ValueError('Dropout parameter exceeded! Must be a value between 0 and 1.')

        for i in range(0, lstm_layers - 1):  # If lstm_layers == 1, this for loop logic is skipped.
            if stateful:
                if i == 0:
                    self.__model.add(ke.layers.LSTM(X.shape[2],
                                                    batch_input_shape=(X.shape[0],
                                                                       X.shape[1],
                                                                       X.shape[2]),
                                                    return_sequences=True,
                                                    stateful=stateful))
                else:
                    self.__model.add(ke.layers.LSTM(X.shape[2],
                                                    input_shape=(X.shape[1],
                                                                 X.shape[2]),
                                                    return_sequences=True,
                                                    stateful=stateful))
            else:
                self.__model.add(ke.layers.LSTM(X.shape[2],
                                                input_shape=(X.shape[1],
                                                             X.shape[2]),
                                                return_sequences=True,
                                                stateful=stateful))
            self.__model.add(ke.layers.Dropout(dropout))
        if lstm_layers > 1:
            self.__model.add(ke.layers.LSTM(X.shape[2],
                                            input_shape=(X.shape[1],
                                                         X.shape[2]),
                                            stateful=stateful,
                                            return_sequences=False))
        else:
            if stateful:
                self.__model.add(ke.layers.LSTM(X.shape[2],
                                                batch_input_shape=(X.shape[0],
                                                                   X.shape[1],
                                                                   X.shape[2]),
                                                stateful=stateful,
                                                return_sequences=False))
            else:
                self.__model.add(ke.layers.LSTM(X.shape[2],
                                                input_shape=(X.shape[1],
                                                             X.shape[2]),
                                                stateful=stateful,
                                                return_sequences=False))
        self.__model.add(ke.layers.Dropout(dropout))
        self.__model.add(ke.layers.Dense(y.shape[1],
                                         kernel_initializer=initializer,
                                         activation=activation))
        self.__model.compile(loss=loss_func, optimizer=optimizer, metrics=['acc'])
        print(self.__model.summary())

    def fit_model(self, X_train=None, X_test=None, y_train=None, y_test=None, epochs=50, batch_size=50, verbose=2,
                  shuffle=False, plot=False):
        """
        Fit data to model & validate. Trains a number of epochs.

        :param: X_train    - (Numpy 2D Array) Numpy matrix consisting of input training features
        :param: X_test     - (Numpy 2D Array) Numpy matrix consisting of input validation/testing features
        :param: y_train    - (Numpy 2D Array) Numpy matrix consisting of output training labels
        :param: y_test     - (Numpy 2D Array) Numpy matrix consisting of output validation/testing labels
        :param: epochs     - (Integer) Integer value denoting number of trained epochs
        :param: batch_size - (Integer) Integer value denoting LSTM training batch_size
        :param: verbose    - (Integer) Integer value denoting net verbosity (Amount of information shown to user during LSTM training)
        :param: shuffle    - (Bool) Boolean value denoting whether or not to shuffle data. This parameter must always remain 'False' for time series datasets.
        :param: plot       - (Bool) Boolean value denoting whether this function should plot out it's evaluation

        :return: None
        """
        if X_test is not None and y_test is not None:
            history = self.__model.fit(x=X_train,
                                       y=y_train,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       validation_data=(X_test, y_test),
                                       verbose=verbose,
                                       shuffle=shuffle)
        else:
            history = self.__model.fit(x=X_train,
                                       y=y_train,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       verbose=verbose,
                                       shuffle=shuffle)

        if plot:
            plt.rcParams['figure.figsize'] = [20, 15]
            plt.plot(history.history['acc'], label='train')
            plt.plot(history.history['val_acc'], label='validation')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

    def predict(self, X, batch_size):
        """
        Predicts label/s from input feature 'X'
        :param: X - Numpy matrix consisting of a single feature vector
        :param: batch_size - (Integer) Denotes prediction batch size
        :return: Numpy matrix of predicted label output
        """
        yhat = self.__model.predict(X, batch_size=batch_size)
        return yhat

    def evaluate(self, y, yhat, plot=False):
        """
        Receives 2D matrix of input features and 2D matrix of output labels, and evaluates input data and target predictions.
        :param: y    - Numpy array consisting of output label vectors (Test Set)
        :param: yhat - Numpy array consisting of output label vectors (Prediction Set)
        :param: plot     - (Bool) Boolean value denoting whether this function should plot out it's evaluation
        :return: None
        """
        y = BinClass.discretize_value(y, 2)
        yhat = BinClass.discretize_value(yhat, 2)
        y = y.flatten()
        yhat = yhat.flatten()

        # F1-Score Evaluation
        print(y)
        print(yhat)
        accuracy = accuracy_score(y, yhat)
        f1 = f1_score(y,
                      yhat,
                      average='macro')  # Calculate metrics globally by counting the total true positives, false negatives and false positives.
        print('Accuracy [' + str(accuracy) + ']')
        print('FScore [' + str(f1) + ']')

        if plot:
            plt.rcParams['figure.figsize'] = [20, 15]
            plt.plot(y, label='actual')
            plt.plot(yhat, label='predicted')
            plt.legend(['actual', 'predicted'], loc='upper left')
            plt.title('Actual vs Predicted')
            plt.show()
        else:
            return accuracy, f1

    @staticmethod
    def write_results_to_disk(path, iteration, lag, test_split, batch, dropout, epoch, layer, activation, initializer,
                              stateful, rmse, accuracy, f_score, time_train):
        """
        Static method which is used for test harness utilities. This method attempts a grid search across many
        trained LSTM models, each denoted with different configurations.

        Attempted configurations:
        * Varied data test split
        * Varied batch sizes
        * Varied epoch counts

        Each configuration is denoted with a score, and used to identify the most optimal configuration.

        :param: path       - (String) String denoting result csv output.
        :param: iteration  - (Integer) Integer denoting test iteration (Unique per test configuration).
        :param: lag        - (Integer) Denotes lag time shift
        :param: test_split - (Float) Float denoting data sample sizes.
        :param: batch      - (Integer) Integer denoting LSTM batch size.
        :param: epoch      - (Integer) Integer denoting number of LSTM training iterations.
        :param: layer      - (Integer) Integer denoting number of LSTM layers
        :param: activation - (String) String denoting activation for LSTM layers.
        :param: initializer- (String) String denoting LSTM initializing weights.
        :param: stateful   - (Bool) Boolean flag which denotes whether LSTM model is trained in stateful mode or not.
        :param: dropout    - (Float) Float denoting model dropout layer.
        :param: rmse       - (Float) Float denoting experiment configuration RSME score.
        :param: accuracy   - (Float) Float denoting experiment accuracy score.
        :param: fscore     - (Float) Float denoting experiment fscore score.
        :param: time_train - (Integer) Integer denoting number of seconds taken by LSTM training iteration.

        :return: None
        """
        file_exists = os.path.isfile(path)
        with open(path, 'a+') as csvfile:
            headers = ['iteration', 'test_split', 'batch', 'epoch', 'layer', 'stateful', 'dropout', 'activation',
                       'initializer', 'rmse', 'accuracy', 'f_score', 'time_train', 'lag']
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
            writer.writerow({'iteration': iteration,
                             'test_split': test_split,
                             'batch': batch,
                             'epoch': epoch,
                             'layer': layer,
                             'stateful': stateful,
                             'dropout': dropout,
                             'activation': activation,
                             'initializer': initializer,
                             'rmse': rmse,
                             'accuracy': accuracy,
                             'f_score': f_score,
                             'time_train': time_train,
                             'lag': lag})

    @staticmethod
    def lag_multiple(X, lag):
        """
        Divides the total number of rows by the lag value, until a perfect multiple amount is retrieved.
        :param X: (Numpy) 2D array consisting of input.
        :param lag: (Integer) Denotes time shift value.
        :return: (Numpy) 2D array consisting of a perfect lag multiple rows.
        """
        n_rows = X.shape[0]
        multiple = int(n_rows/lag)
        max_new_rows = multiple * lag
        return X[0:max_new_rows, :]


""" Grid Search """

for test_split in test_harness_param:
    X_train, X_validate, y_train, y_validate = train_test_split(X_df, y_df, test_size=test_split)
    X_train = X_train.values
    y_train = y_train.values
    X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=.5)
    X_validate = X_validate.values
    y_validate = y_validate.values
    X_test = X_test.values
    y_test = y_test.values

    # Lag Multiples
    X_train = LSTM.lag_multiple(X=X_train, lag=lag)
    y_train = LSTM.lag_multiple(X=y_train, lag=lag)
    X_validate = LSTM.lag_multiple(X=X_validate, lag=lag)
    y_validate = LSTM.lag_multiple(X=y_validate, lag=lag)
    X_test = LSTM.lag_multiple(X=X_test, lag=lag)
    y_test = LSTM.lag_multiple(X=y_test, lag=lag)

    # Reshape for fitting in LSTM
    X_train = X_train.reshape((int(X_train.shape[0] / lag), lag, X_train.shape[1]))
    y_train = y_train[0:int(y_train.shape[0] / lag),:]
    X_validate = X_validate.reshape((int(X_validate.shape[0] / lag), lag, X_validate.shape[1]))
    y_validate = y_validate[0:int(y_validate.shape[0] / lag),:]
    X_test = X_test.reshape((int(X_test.shape[0] / lag), lag, X_test.shape[1]))
    y_test = y_test[0:int(y_test.shape[0] / lag),:]

    # print('\nReshaping Training Frames')
    # print("X_train shape [" + str(X_train.shape) + "] Type - " + str(type(X_train)))
    # print("y_train shape [" + str(y_train.shape) + "] Type - " + str(type(y_train)))
    # print("X_validate shape [" + str(X_validate.shape) + "] Type - " + str(type(X_validate)))
    # print("y_validate shape [" + str(y_validate.shape) + "] Type - " + str(type(y_validate)))
    # print("X_test shape [" + str(X_test.shape) + "] Type - " + str(type(X_test)))
    # print("y_test shape [" + str(y_test.shape) + "] Type - " + str(type(y_test)))

    for epochs in max_epochs:
        for batch in max_batch:
            for activation in activations:
                for layer in layers:
                    for dropout in drop_out:
                        for initializer in initializers:
                            t0 = time.time()
                            # Train on discrete data (Train > Validation)
                            model = LSTM(X=X_train,
                                         y=y_train,
                                         lag=lag,
                                         loss_func='binary_crossentropy',
                                         activation=activation,
                                         optimizer='adam',
                                         lstm_layers=layer,
                                         dropout=dropout,
                                         stateful=state,
                                         initializer=initializer)
                            model.fit_model(X_train=X_train,
                                            X_test=X_validate,
                                            y_train=y_train,
                                            y_test=y_validate,
                                            epochs=epochs,
                                            batch_size=batch,
                                            verbose=2,
                                            shuffle=False,
                                            plot=False)

                            acc_list, f_list = [], []
                            for i in range(0, X_validate.shape[0]):
                                X = np.array(np.array(X_validate[i,:]))
                                X = X.reshape((int(X.shape[0] / lag), lag, X.shape[1]))
                                y = model.predict(X, batch_size=batch)
                                model.fit_model(X_train=X,
                                                y_train=y,
                                                epochs=5,
                                                batch_size=1,
                                                verbose=1,
                                                shuffle=False,
                                                plot=False) # Online Learning, Training on validation predictions.
                                acc_score, f_score = model.evaluate(y=y_validate[i, :],
                                                                    yhat=y,
                                                                    plot=False)
                                acc_list.append(acc_score)
                                f_list.append(f_score)

                            t1 = time.time()
                            time_total = t1 - t0
                            LSTM.write_results_to_disk(path="query_sequence_lstm_results.csv",
                                                       iteration=iteration,
                                                       lag=lag,
                                                       test_split=test_split,
                                                       epoch=epochs,
                                                       layer=layer,
                                                       dropout=dropout,
                                                       stateful=state,
                                                       batch=batch,
                                                       activation=activation,
                                                       initializer=initializer,
                                                       rmse=None,
                                                       accuracy=sum(acc_list) / len(acc_list),
                                                       f_score=sum(f_list) / len(f_list),
                                                       time_train=time_total)
                            print('----------------------------' + str(iteration) + '----------------------------')
                            iteration += 1