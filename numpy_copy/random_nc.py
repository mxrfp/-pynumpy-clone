import random as rn
from numpy_copy import numpy_copy_better as nc

def random_array(shape, dtype: nc.dtypes = float) -> nc.Array:
    return nc.Array(nc.like_shape(shape, rn.random, dtype=dtype))



