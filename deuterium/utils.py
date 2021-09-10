from symengine import symbols
from .autograd import Variable
import numpy as np


def random_symbols(n, prefix):
    """Generate n random symbols of the form prefix_{number}"""
    ret = []
    for subscript in range(n):
        word = f"{prefix}_{subscript}"
        ret.append(symbols(word))
    return ret


to_vec = np.vectorize(lambda x: Variable(x))

