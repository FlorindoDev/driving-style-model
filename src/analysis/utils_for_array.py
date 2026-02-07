import numpy as np


def avg_in_window(arr, i, win):
    start = max(0, i - win)
    end = min(len(arr), i + win + 1)
    vals = [abs(a) for a in arr[start:end]]
    return sum(vals) / len(vals)

def find_frist_value(array, value):
    #primo indice dove array[i] >= value
    for i, d in enumerate(array):
        if d >= value:
            return i
    return len(array) - 1

#Ritorna una indice
def find_closest_value(array, value):
    array = np.asarray(array)
    return np.abs(array - value).argmin()

def find_last_value(array, value):
    #ultimo indice dove array[i] <= value
    last = 0
    for i, d in enumerate(array):
        if d <= value:
            last = i
        else:
            break
    return last