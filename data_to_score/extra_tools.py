import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def normalize_data(df, column_name):
    """Normalize a column in a pandas DataFrame using median and MAD scaling.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the column to normalize.
        column_name (str): The name of the column to normalize.
    
    Returns:
        tuple: A tuple of two numpy arrays: (X_train, X_test).
    """
    # Split the data into train and test sets
    train_size = int(0.7 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]

    # Compute the median and MAD on the train data
    median = np.median(train_data[column_name])
    mad = np.median(np.abs(train_data[column_name] - median))

    # Normalize the train and test data separately
    train_data[column_name] = (train_data[column_name] - median) / mad
    test_data[column_name] = (test_data[column_name] - median) / mad

    # Extract the normalized column data as numpy arrays
    X_train = train_data[column_name].to_numpy().reshape(-1, 1)
    X_test = test_data[column_name].to_numpy().reshape(-1, 1)

    return X_train, X_test

def df_percentage_of_ones(df, column_name):
    num_ones = df[column_name].sum()
    total = df[column_name].count()
    percent_ones = (num_ones / total)
    return percent_ones

def arr_percentage_of_ones(array):
    num_ones = np.count_nonzero(array == 1)
    total = array.size
    percent_ones = (num_ones / total) 
    return percent_ones



import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from ipywidgets import interactive

import matplotlib.pyplot as plt
import numpy as np

def plot_anomalies(time_series, labels1, labels2):
    """
    Plots a time series with two sets of binary labels indicating whether each point is anomalous or not.
    Anomalous points in each set of labels are plotted in different colors.
    Allows the user to interactively scroll through the data.
    
    Args:
    - time_series: numpy array representing the time series
    - labels1: numpy array representing the first set of binary labels
    - labels2: numpy array representing the second set of binary labels
    
    Returns:
    None
    """
    
    # Create a figure and axes for the plot
    fig, ax = plt.subplots()
    
    # Set up the plot for the time series
    ax.plot(time_series[:,0], color='black', label='Time Series')
    
    # Add the first set of labels to the plot
    mask1 = labels1[:,0].astype(bool)
    ax.scatter(np.arange(len(time_series))[mask1], time_series[:,0][mask1], 
               color='red', label='true_anomalies')
    
    # Add the second set of labels to the plot
    mask2 = labels2[:,0].astype(bool)
    ax.scatter(np.arange(len(time_series))[mask2], time_series[:,0][mask2], 
               color='blue', label='predicted')
    
    # Set up the legend for the plot
    ax.legend(loc='upper left')
    
    # Set up the interactive scrolling for the plot
    ax.set_xlim([0, len(time_series)])
    ax.set_ylim([np.min(time_series), np.max(time_series)])
    ax.set_title('Time Series with Anomalies')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    
    plt.show()


def roll_window(time_series, window_size):
    num_windows = len(time_series) - window_size + 1
    rolled_array = np.zeros((num_windows, window_size))
    
    for i in range(num_windows):
        rolled_array[i] = time_series[i:i+window_size].flatten()

    return rolled_array



def roll_binary_window(dataframe, column_name, window_size):
    time_series = dataframe[column_name].to_numpy()
    num_windows = len(time_series) - window_size + 1
    rolled_array = np.zeros(num_windows, dtype=np.int)

    for i in range(num_windows):
        if np.any(time_series[i:i+window_size] == 1):
            rolled_array[i] = 1

    output_dataframe = pd.DataFrame(rolled_array, columns=[column_name])
    return output_dataframe





