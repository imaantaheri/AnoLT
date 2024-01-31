import numpy as np

def roll_window(binary_list, window_size):
    binary_array = np.array(binary_list)
    rolling_array = np.lib.stride_tricks.sliding_window_view(binary_array, window_shape=window_size)
    result = np.any(rolling_array, axis=1).astype(int)
    
    return result.tolist()


