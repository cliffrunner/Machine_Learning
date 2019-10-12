import numpy as np
def build_space(lower, upper, step, binary=True):
    if binary:
        return [bin(x)[2:].zfill(20) for x in np.arange(lower, upper, step)]
    else:
        return np.arange(lower, upper, step)