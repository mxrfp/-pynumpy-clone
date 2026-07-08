from __future__ import annotations
from fractions import Fraction
from math import nan
import sys
from typing import Any


sys.setrecursionlimit(10**7)

dtypes =  type[float] | type[complex] | type[int] 

def is_valid(self):
    if self == []:
        return True
    if not isinstance(self, list):
        return True 
    is_True = True
    if isinstance(self[0], list):
        base_len = len(self[0])
    base_type = type(self[0])
    for i in self:
        if base_type!=list and type(i)!=list:
            is_True = True
        elif base_type==list and type(i) == list:
            is_True = True
        else:
            is_True = False
        if not isinstance(i, list):
            continue
        is_True = is_True and (base_len == len(i)) #type: ignore
    return is_True and (not (False in [is_valid(i) for i in self]))


def shape(self):
        if self == []:
            return []
        if not isinstance(self[0], list):
            return [len(self)]
        return [len(self)] + shape(self[0])

def complex_real(num):
    return complex(num,0)

mask={
    complex: complex_real,
    int:int,
    float:float
}

def like_shape(shape, func, dtype: dtypes = float):
    if len(shape) == 1:
        return [mask[dtype](func()) for _ in range(shape[0])]
    return [like_shape(shape[1:], func, dtype=dtype) for _ in range(shape[0])]

def dtype(self):
    def flatten(lst):
        if not lst:
            return []
        if not isinstance(lst[0], list):
            return [type(lst[0])] + flatten(lst[1:])
        return flatten(lst[0]) + flatten(lst[1:]) 
    dtype_lst = flatten(self)
    base_dtype = dtype_lst[0]
    for i in dtype_lst[1:]:
        if base_dtype != i:
            raise ValueError("Cannot create an array with different dtypes.")
    else:
        return base_dtype
    
    
class Array:
    def __init__(self, lst):
        self.body = lst 
        self.valid_state = is_valid(lst)
        self.shape = shape(lst) if self.valid_state else None
        self.ndim = len(self.shape) if self.shape else None
        self.size = int(self.valid_state) if self.ndim is not None else 0
        self.dtype = dtype(lst)
        if self.shape:
            for i in self.shape:
                self.size *= i
    def __str__(self) -> str:
        return ("array(" + str(self.body) +")")
    class Decide:
        def __init__(self, function):
            self.function = function
            self.python_method = False
        def decide_type(self, n, other):
            d = {
                "__add__": {
                    float: float.__add__,
                    list: list.__add__,
                    complex: complex.__add__,
                    int: int.__add__
                    },
                "__sub__": {
                    int: int.__sub__,
                    float: float.__sub__,
                    complex: complex.__sub__
                },
                "__mul__": {
                    int: int.__mul__,
                    float: float.__mul__,
                    complex: complex.__mul__
                },
                "__truediv__": {
                    int: int.__truediv__,
                    float: float.__truediv__,
                    complex: complex.__truediv__
                },
                "__floordiv__": {
                    int: int.__floordiv__,
                    float: float.__floordiv__,
                },
                "__mod__": {
                    int: int.__mod__,
                    float: float.__mod__
                },
                "__pow__": {
                    int: int.__pow__,
                    float: float.__pow__,
                    complex: complex.__pow__
                },
                "__abs__":{
                    int: int.__abs__,
                    float: float.__abs__,
                    complex: complex.__abs__
                }
            }
            if other is None:
                return d[self.function][type(n)]
            if self.function not in d:
                return self.function
            if type(n) != type(other):
                raise TypeError(f"Cannot apply {self.function} to arrays with different dtypes or ndim.")
            if type(n) not in d[self.function]:
                raise ValueError(f"Type {type(n)} is not supported for broadcasting with function: {self.function}")
            if type(other) not in d[self.function]:
                raise ValueError(f"Type {type(other)} is not supported for broadcasting with function: {self.function}")
            
            return d[self.function][type(other)]
        def is_python(self, array, other):
            if isinstance(other, Array):
                if not (other.valid_state  and other.shape == array.shape and other.ndim is not None):
                    self.python_method = True
        def decide_other(self, array, other):
            if not isinstance(other, Array):
                if array.body == []:
                    raise ValueError(f"Cannot broadcast a {type(other)} in an empty array.")
                return Array.alike(array, other)
            return other
    def update(self)-> None:
        self.valid_state = is_valid(self.body)
        self.shape = shape(self.body) if self.valid_state else None
        self.ndim = len(self.shape) if self.shape else None
        self.size = bool(self.valid_state)
        if self.shape:
            for i in self.shape:
                self.size *= i
        self.dtype = dtype(self.body)
    def vectorize(self, o, d) -> Array:
        def helper(lst, other, decider):
            if decider.python_method:
                if decider.function == "__add__":
                    return lst+other
                raise ValueError(f"Cannot {decider.function} a non-valid array.")
            if lst == []:
                return []
            if not isinstance(lst[0], list) and not isinstance(other[0], list):
                new_l = []
                for i, j in zip(lst, other):
                    if j == 0 and decider.function in {"__truediv__", "__floordiv__"}:
                        new_l.append(nan)
                    else:
                        new_l.append(decider.decide_type(i,j)(i,j))
                return new_l
            new_l = []
            for i, j in zip(lst, other):
                new_l.append(helper(i, j, decider))
            return new_l
        found = helper(self.body, o.body, d)
        return Array(found)
    def __add__(self, other) -> Array:
        decider = self.Decide("__add__")
        decider.is_python(self, other)
        return Array.vectorize(self, decider.decide_other(self, other), decider)
    def __mod__(self, other):
        decider = self.Decide("__mod__")
        decider.is_python(self, other)
        return Array.vectorize(self, decider.decide_other(self, other), decider)
    def __radd__(self, other) -> Array:
        return self + other
    def __mul__(self, other) -> Array:
        decider = self.Decide("__mul__")
        decider.is_python(self, other)
        return Array.vectorize(self, decider.decide_other(self, other), decider)
    def __truediv__(self, other) -> Array:
        decider = self.Decide("__truediv__")
        decider.is_python(self, other)
        return Array.vectorize(self, decider.decide_other(self, other), decider)
    def __rtruediv__(self, other) -> Array:
        return self/other
    def __floordiv__(self, other) -> Array:
        decider = self.Decide("__floordiv__")
        decider.is_python(self, other)
        return Array.vectorize(self, decider.decide_other(self, other), decider)
    def __rfloordiv__(self, other) -> Array:
        return self//other
    def __rmul__(self, other) -> Array:
        return self*other
    def __neg__(self) -> Array:
        return self*(self.dtype(-1))
    def __sub__(self, other) -> Array:
        return self + (-other)
    def __pos__(self) -> Array:
        return self
    def alike(self, num) -> Array:
        def helper(lst: list, n) -> list:
            if lst == []:
                return []
            if not isinstance(lst[0], list):
                return [n for _ in range(len(lst))]
            return [helper(i, n) for i in lst]
        return Array(helper(self.body, num))
    def zeros(self) -> Array:
        return Array.alike(self, 0)
    def tolist(self):
        return self.body
    def __len__(self) -> int:
        return len(self.body)
    def __getitem__(self, other, brd=False)  -> Any:
        if brd:
            if not other:
                return self
            if not self:
                return []
            if isinstance(self[0], int | float | complex):
                return self[other[0]]
            if isinstance(self[0], list):
                return [Array.__getitem__(self[0][other[0]], other[1:], brd=True)] +  \
                    Array.__getitem__(self[1:], other, brd=True) #type: ignore
        if isinstance(other, int | slice):
            return Array(self.body[other])
        return Array(Array.__getitem__(self.body[other[0]], other[1:], brd=True))
    def __setitem__(self, key, val):
        if isinstance(val, Array):
            val = val.body
        self.body[key] = val
        self.update()
    def self_function(self, decieder) -> Array | list:
        if not isinstance(self.body[0], list):
            return [decieder.decide_type(i, None)(i) for i in self.body]
        return Array([i.self_function(decieder) for i in self])
    def __abs__(self):
        decider = self.Decide("__abs__")
        return self.self_function(decider)
    def __matmul__(self, other):
        if not isinstance(other, Array):
            raise TypeError(f"Other has to be an Array, not {type(other)}")
    def __array__(self):
        try:
            import numpy
        except ModuleNotFoundError:
            raise ModuleNotFoundError("numpy is necessary to plot the array")
        return numpy.array(self.body)
        

def linspace(start, stop, points, dtype: dtypes = float) -> Array:
    start = Fraction(start)
    stop = Fraction(stop)
    step = Fraction(stop-start, points-1) 
    arr = [mask[dtype](start)]
    current = start

    for _ in range(points-1):
        current = Fraction(current, 1) + step
        arr.append(mask[dtype](current))

    return Array(arr)

def meshgrid(x, y) -> Array:
    x_arr = [x.tolist() for _ in range(len(y))]
    y_arr_or = [y.tolist() for _ in range(len(x))]
    y_arr = [i.copy() for i in y_arr_or]

    for yc in range(len(y_arr_or)):
        for xc in range(len(y)):
            y_arr[yc][xc] = y_arr_or[xc][len(y_arr)-1-yc]

    return Array([x_arr, y_arr])



def zeros(shape, dtype: dtypes = float) -> Array:
    def return_0():
        return 0
    return Array(like_shape(shape, return_0, dtype=dtype))

def arange(stop, start=0, step=1, dtype: dtypes = float) -> Array:
    start, stop = sorted((start, stop))
    new_lst = list(range(start, stop, step))
    new_dtype_lst = []
    for i in new_lst:
        new_dtype_lst.append(mask[dtype](i))
    return Array(new_dtype_lst)


