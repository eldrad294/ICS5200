import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import pandas as pd
import statsmodels
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
import sklearn as sk
import theano
import tensorflow
import plaidml.keras
plaidml.keras.install_backend()
import keras as ke
import math
import csv
import os.path
import time

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
    def discretize_value(df, n):
        """
        param: df - Input data
        param: n - Number of buckets
        """
        BinClass.__validate(df, n)
        for column in df.columns:
            max_val = df[column].max(skipna=True)
            threshold = max_val / n
            df[column] = df[column].apply(lambda x: BinClass.__bucket_val(x, threshold, n))
            if df[column].isnull().values.any():
                df[column] = df[column].fillna(1)
        return df


# LSTM Class
class LSTM:
    """
    Long Short Term Memory Neural Net Class
    """

    def __init__(self, X, y, loss_func, activation, mode='regression', optimizer='sgd', lstm_layers=1, dropout=.0):
        """
        Initiating the class creates a net with the established parameters
        :param X             - (Numpy 2D Array) Training data used to train the model (Features).
        :param y             - (Numpy 2D Array) Test data used to test the model (Labels).
        :param loss_function - (String)  Denotes mode of measure fitting of model (Fitting function).
        :param activation    - (String)  Neuron activation function used to activate/trigger neurons.
        :param mode          - (String)  A flag used to set specific model training mode (Classification OR Regression).
        :param optimizer     - (String)  Denotes which function to us to optimize the model build (eg: Gradient Descent).
        :param lstm_layers   - (Integer) Denotes the number of LSTM layers to be included in the model build.
        :param dropout       - (Float)   Denotes amount of dropout for model. This parameter must be a value between 0 and 1.
        """
        self.mode = mode
        self.model = ke.models.Sequential()
        for i in range(0, lstm_layers - 1):  # If lstm_layers == 1, this for loop logic is skipped.
            self.model.add(ke.layers.LSTM(X_train.shape[2], input_shape=(X_train.shape[1], X_train.shape[2]),
                                          return_sequences=True))
        self.model.add(ke.layers.LSTM(X.shape[2], input_shape=(X.shape[1], X.shape[2])))
        if dropout > 1 and droptout < 0:
            raise ValueError('Dropout parameter exceeded! Must be a value between 0 and 1.')
        self.model.add(ke.layers.Dropout(dropout))
        self.model.add(ke.layers.Dense(y.shape[1]))
        self.model.add(ke.layers.Activation(activation.lower()))
        self.model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])
        print(self.model.summary())

    def fit_model(self, X_train, X_test, y_train, y_test, epochs=50, batch_size=50, verbose=2, shuffle=False,
                  plot=False):
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

    def predict(self, X):
        """
        Predicts label/s from input feature 'X'

        :param: X - Numpy matrix consisting of a single feature vector

        :return: Numpy matrix of predicted label output
        """
        yhat = self.model.predict(X)
        return yhat

    def predict_and_evaluate(self, X, y, y_labels, plot=False, category=1):
        """
        Receives 2D matrix of input features and 2D matrix of output labels, and evaluates input data and target predictions.

        :param: X        - (Numpy 2D Array) Numpy array consisting of input feature vectors
        :param: y        - (Numpy 2D Array) Numpy array consisting of target/output labels
        :param: y_labels - (List) List of target label names
        :param: plot     - (Bool) Boolean value denoting whether this function should plot out it's evaluation
        :param: category - (Integer) Integer value denoting category type to use during evaluation. This parameter is
                           experiment specific, refer to below. Note, that this parameter should always be
                           established as a value of 1.

        :return: None
        """
        yhat = self.predict(X)

        # RMSE Evaluation
        if self.mode == 'regression':
            rmse = math.sqrt(mean_squared_error(y, yhat))
            if not plot:
                return rmse
            print('Reported: ' + str(rmse) + ' rmse')

        elif self.mode == 'classification':
            column_names = []
            for i in range(len(y_labels) * lag):
                column_names.append("column" + str(i))

            # Denote category type
            if category == 1:
                y = pd.DataFrame(y, columns=column_names)
                y = BinClass.discretize_value(y, bin_value)
                y = y.values
            yhat = pd.DataFrame(yhat, columns=column_names)
            yhat = BinClass.discretize_value(yhat, bin_value)
            yhat = yhat.values

            # F1-Score Evaluation
            for i in range(len(y_labels)):
                accuracy = accuracy_score(y[:, i], yhat[:, i])
                f1 = f1_score(y[:, i],
                              yhat[:, i],
                              average='micro')
                print('Accuracy [' + y_labels[i] + ']: ' + str(accuracy))
                print('FScore [' + y_labels[i] + ']: ' + str(f1))

        if plot:
            for i in range(0, len(y[0])):
                plt.rcParams['figure.figsize'] = [20, 15]
                plt.plot(y[:, i], label='actual')
                plt.plot(yhat[:, i], label='predicted')
                plt.legend(['actual', 'predicted'], loc='upper left')
                plt.title(y_labels[i % len(y_labels)] + " +" + str(math.ceil((i + 1) / len(y_label))))
                plt.show()

    @staticmethod
    def write_results_to_disk(path, iteration, lag, test_split, batch, depth, dropout, epoch, score, time_train):
        """
        Static method which is used for test harness utilities. This method attempts a grid search across many
        trained LSTM models, each denoted with different configurations.

        Attempted configurations:
        * Varied lag projection
        * Varied data test split
        * Varied batch sizes
        * Varied LSTM neural depth
        * Varied epoch counts

        Each configuration is denoted with a score, and used to identify the most optimal configuration.

        :param: path       - (String) String denoting result csv output.
        :param: iteration  - (Integer) Integer denoting test iteration (Unique per test configuration).
        :param: lag        - (Integer) Integer denoting lag value.
        :param: test_split - (Float) Float denoting data sample sizes.
        :param: batch      - (Integer) Integer denoting LSTM batch size.
        :param: depth      - (Integer) Integer denoting LSTM neuron depth size.
        :param: epoch      - (Integer) Integer denoting number of LSTM training iterations.
        :param: dropout    - (Float) Float denoting model dropout layer.
        :param: score      - (Float) Float denoting experiment configuration RSME score.
        :param: time_train - (Integer) Integer denoting number of seconds taken by LSTM training iteration.

        :return: None
        """
        file_exists = os.path.isfile(path)
        with open(path, 'a+') as csvfile:
            headers = ['iteration', 'lag', 'test_split', 'batch', 'depth', 'epoch', 'dropout', 'score', 'time_train']
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header
            writer.writerow({'iteration': iteration,
                             'lag': lag,
                             'test_split': test_split,
                             'batch': batch,
                             'depth': depth,
                             'epoch': epoch,
                             'dropout': dropout,
                             'score': score,
                             'time_train': time_train})

# Test Harness
iteration = 0
max_lag_param = 15
# Training / Test Split Subsets. Test subset is split 50/50 for validation/test
test_harness_param = (.5, .6, .7, .8, .9)
batch = (10, 25, 50, 75, 100)
epochs = (50, 100, 150)
lstm_depth = (1, 2, 3)
dropout = (.1, .2, .3, .4, .5)

# Test Multiple Time Splits (Lag)
for lag in range(1, max_lag_param+1):

    t0 = time.time()  # Capture Time Shot

    shifted_df = series_to_supervised(df, lag, lag)  # Shifting both in past and future

    # Separate labels from features
    y_df_column_names = shifted_df.columns[len(df.columns):len(df.columns) + len(y_label)]
    y_df = shifted_df[y_df_column_names]
    X_df = shifted_df.drop(columns=y_df_column_names)

    # Test Multiple Train/Validation Splits
    for test_split in test_harness_param:
        X_train, X_validate, y_train, y_validate = train_test_split(X_df, y_df, test_size=test_split)
        X_train = X_train.values
        y_train = y_train.values
        X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=.5)
        X_validate = X_validate.values
        y_validate = y_validate.values

        # Reshape for fitting in LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_validate = X_validate.reshape((X_validate.shape[0], 1, X_validate.shape[1]))

        for depth in lstm_depth:
            for drop in dropout:
                for epoch in epochs:
                    for batch_size in batch:

                        # Build model
                        model = LSTM(X=X_train,
                                     y=y_train,
                                     loss_func='categorical_crossentropy',
                                     activation='sigmoid',
                                     optimizer='adam',
                                     mode='classification',
                                     lstm_layers=depth,
                                     dropout=drop)

                        # Train model
                        model.fit_model(X_train=X_train,
                                        X_test=X_validate,
                                        y_train=y_train,
                                        y_test=y_validate,
                                        epochs=epoch,
                                        batch_size=batch_size,
                                        verbose=2,
                                        shuffle=False,
                                        plot=False)

                        # Evaluate the model
                        rmse = model.predict_and_evaluate(X=X_validate,
                                                          y=y_validate,
                                                          y_labels=y_label,
                                                          plot=False)

                        t1 = time.time()
                        time_total = t1 - t0

                        # Write results to disk
                        LSTM.write_results_to_disk(path="time_series_lstm_regression_results.csv",
                                                   iteration=iteration,
                                                   lag=lag,
                                                   test_split=test_split,
                                                   batch=batch_size,
                                                   depth=depth,
                                                   epoch=epoch,
                                                   dropout=drop,
                                                   score=str(rmse),
                                                   time_train=time_total)

                        iteration += 1
