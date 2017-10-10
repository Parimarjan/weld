from weldarray import *
from weldnumpy import *

# importing everything from numpy so we can selectively over-ride the array creation routines, and
# let other functions go to numpy.
from numpy import *

# Need this for things like wrapping np.zeros, np.array etc. Or alternatively, could use decorators as
# with random.
import numpy as np

import numpy.random as random
'''
Using decorators to add a wrapper class around all functions in the np.random class.
'''
import functools
def weldarray_decorator(f):
    '''
    Accepts an arbitrary function, and wraps the return value to be a weldarray whenever
    appropriate.
    '''
    # it still behaves the same way without functools.wraps, but this helps with debugging - so the
    # wrapped function will have documentation / and name as the original function.
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ret = f(*args, **kwargs)
        if isinstance(ret, np.ndarray) and str(ret.dtype) in SUPPORTED_DTYPES:
            print('ret: ', ret)
            print('returning weldarray')
            a = weldarray(ret)
            print(type(a))
            return a

        return ret
    return wrapper

def decorate_all_in_module(module, decorator):
    for name in dir(module):
        obj = getattr(module, name)
        # Not all callables are functions - this seems to be especially true in the random class.
        if callable(obj):
            setattr(module, name, decorator(obj))

# so weldnumpy.random will behave as np.random with a wrapper.
decorate_all_in_module(random, weldarray_decorator)

'''
Array Creation Routines: Since there are only a few array creation routines, it seems
simpler to just make wrapper functions for each of them rather than use a decorator.
'''

def array(arr, *args, **kwargs):
    '''
    Wrapper around weldarray - first create np.array and then convert to
    weldarray.
    '''
    return weldarray(np.array(arr, *args, **kwargs))

def zeros(*args, **kwargs):
    return weldarray(np.zeros(*args, **kwargs))

def zeros_like(*args, **kwargs):
    return weldarray(np.zeros_like(*args, **kwargs))

def ones(*args, **kwargs):
    return weldarray(np.ones(*args, **kwargs))

def ones_like(*args, **kwargs):
    return weldarray(np.ones_like(*args, **kwargs))

def full(*args, **kwargs):
    return weldarray(np.full(*args, **kwargs))

def full_like(*args, **kwargs):
    return weldarray(np.full_like(*args, **kwargs))

def empty(*args, **kwargs):
    return weldarray(np.empty(*args, **kwargs))

def empty_like(*args, **kwargs):
    return weldarray(np.empty_like(*args, **kwargs))

def eye(*args, **kwargs):
    return weldarray(np.eye(*args, **kwargs))

def identity(*args, **kwargs):
    return weldarray(np.identity(*args, **kwargs))

# functions that don't exist in numpy
def erf(weldarray):
    '''
    FIXME: This is kinda hacky because all other function are routed through __array_ufunc__ by
    numpy and here we directly call _unary_op. In __array_ufun__ I was using properties of ufuncs,
    like ufunc.__name__, so using that route would require special casing stuff. For now, this is
    just the minimal case to make blackscholes work.
    '''
    return weldarray._unary_op('erf')
