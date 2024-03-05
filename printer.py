import numpy as np

def vector_to_str_pretty(vec:np.ndarray, precision:int):
    """ Converts all vecor entries into strings to enable pretty formatted printing of vectors.
        Positive numbers get a leading ' ' added, to account for no '-' in print. Trailing 0s are also added to all 
        vector components if missing.
    """
    converted = list()
    converted = [str(np.format_float_positional(round(vec[i], precision), trim='-')) for i in range (0, len(vec))]

    for i in range (0, len(vec)):
        if vec[i] >= 0: 
            converted[i] = " " + str(np.format_float_positional(round(vec[i], precision), trim='-'))

        if len(converted[i]) < precision + 3:
            converted[i] += "0"*(precision + 3 - len(converted[i]))               
    return converted