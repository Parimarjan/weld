import numpy as np
import py.test
import random
from weldnumpy import *

'''
TODO: Set up multi-dimensional array creation routines.
'''

UNARY_OPS = [np.exp, np.log, np.sqrt]
# TODO: Add wa.erf - doesn't use the ufunc functionality of numpy so not doing it for
# now.
BINARY_OPS = [np.add, np.subtract, np.multiply, np.divide]
REDUCE_UFUNCS = [np.add.reduce, np.multiply.reduce]

TYPES = ['float32', 'float64', 'int32', 'int64']
SHAPES = [(2,2), (3,7), (9,1,4), (2,5,7,2)]

# TODO: Create test with all other ufuncs.
def random_arrays(shape, dtype):
    '''
    Generates random Weld array, and numpy array of the given num elements.
    '''
    # np.random does not support specifying dtype, so this is a weird
    # way to support both float/int random numbers
    test = np.zeros((shape), dtype=dtype)
    test[:] = np.random.randn(*test.shape)
    test = np.abs(test)
    # at least add 1 so no 0's (o.w. divide errors)
    random_add = np.random.randint(1, high=10, size=test.shape)
    test = test + random_add
    test = test.astype(dtype)

    np_test = np.copy(test)
    w = weldarray(test, verbose=False)

    return np_test, w

def given_arrays(l, dtype):
    '''
    @l: list.
    returns a np array and a weldarray.
    '''
    test = np.array(l, dtype=dtype)
    np_test = np.copy(test)
    w = weldarray(test)

    return np_test, w

# TODO: Common tests
def test_unary_elemwise():
    '''
    Tests all the unary ops in UNARY_OPS.

    FIXME: For now, unary ops seem to only be supported on floats.
    '''
    for SHAPE in SHAPES:
        for op in UNARY_OPS:
            for dtype in TYPES:
                print(dtype)
                # int still not supported for the unary ops in Weld.
                if "int" in dtype:
                    continue
                np_test, w = random_arrays(SHAPE, dtype)
                w2 = op(w)
                np_result = op(np_test)
                w2_eval = w2.evaluate()

                assert np.allclose(w2, np_result)
                assert np.array_equal(w2_eval, np_result)

def test_binary_elemwise():
    '''
    '''
    for SHAPE in SHAPES:
        for op in BINARY_OPS:
            for dtype in TYPES:
                np_test, w = random_arrays(SHAPE, dtype)
                np_test2, w2 = random_arrays(SHAPE, dtype)
                w3 = op(w, w2)
                weld_result = w3.evaluate()
                np_result = op(np_test, np_test2)
                # Need array equal to keep matching types for weldarray, otherwise
                # allclose tries to subtract floats from ints.
                assert np.array_equal(weld_result, np_result)

def test_mix_np_weld_ops():
    '''
    Weld Ops + Numpy Ops - before executing any of the numpy ops, the
    registered weld ops must be evaluateuated.
    '''
    for SHAPE in SHAPES:
        np_test, w = random_arrays(SHAPE, 'float32')
        np_test = np.exp(np_test)
        np_result = np.sin(np_test)

        w2 = np.exp(w)
        w2 = np.sin(w2)
        weld_result = w2.evaluate()
        assert np.allclose(weld_result, np_result)

def test_scalars():
    '''
    Special case of broadcasting rules - the scalar is applied to all the
    Weldrray members.
    '''
    for SHAPE in SHAPES:
        t = "int32"
        print("t = ", t)
        n, w = random_arrays(SHAPE, t)
        n2 = n + 2
        w2 = w + 2

        w2 = w2.evaluate()
        assert np.allclose(w2, n2)

        # test by combining it with binary op.
        n, w = random_arrays(SHAPE, t)
        w += 10
        n += 10

        n2, w2 = random_arrays(SHAPE, t)

        w = np.add(w, w2)
        n = np.add(n, n2)

        assert np.allclose(w, n)

        t = "float32"
        print("t = ", t)
        np_test, w = random_arrays(SHAPE, t)
        np_result = np_test + 2.00
        w2 = w + 2.00
        weld_result = w2.evaluate()
        assert np.allclose(weld_result, np_result)

def test_shapes():
    '''
    After creating a new array and doing some operations on it - the shape, ndim, and {other
    atrributes?} should remain the same as before.
    '''
    for SHAPE in SHAPES:
        n, w = random_arrays(SHAPE, 'float32')
        print('view: ', w._weldarray_view)
        n = np.exp(n)
        w = np.exp(w)
        print('view: ', w._weldarray_view)
        w = w.evaluate()
        
        assert n.shape == w.shape
        assert n.ndim == w.ndim
        assert n.size == w.size
        assert np.array_equal(w, n)
    

'''
More advanced tests.
'''

'''
Broadcasting based tests.
'''

'''
'''

'''
create new arrays in different ways: transpose, reshape, concatenation, horizontal/vertical etc.
'''

def test_subtle_new_array():
    pass
    '''
    FIXME:
    a.T, or a.imag seem to create new weldarrays without passing through __init__. Might have to
    create __array_finalize__ after all.
    '''
    # for a in attr:
        # if a == 'imag' or a == 'T':
            # continue
        # print(a)
        # print(eval('n.' + a))
        # print(eval('w.' + a))

'''
Views based tests.
'''
# def test_views_non_contig():
    # pass

# def test_views_contig_basic():
    # pass

