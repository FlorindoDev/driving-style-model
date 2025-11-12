# ------------------------------------------------- utilità -------------------------------------------------
def avg_in_window(arr, i, win):
    start = max(0, i - win)
    end = min(len(arr), i + win + 1)
    vals = [abs(a) for a in arr[start:end]]
    return sum(vals) / len(vals)

def find_frist_value(dist_array, value):
    #primo indice dove distance >= value
    for i, d in enumerate(dist_array):
        if d >= value:
            return i
    return len(dist_array) - 1

def find_last_value(dist_array, value):
    #ultimo indice dove distance <= value
    last = 0
    for i, d in enumerate(dist_array):
        if d <= value:
            last = i
        else:
            break
    return last