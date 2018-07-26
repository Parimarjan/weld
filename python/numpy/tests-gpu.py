import weldnumpy as wn
from weldnumpy import weldarray
import numpy as np
import py.test

# FIXME: int's seem to be failing roughly half the tests..
TYPES = [np.float64]
NUM_ELS = 10

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

def test_simple():
    wn.set_nvvm(1)
    for t in TYPES:
        _, w1 = random_arrays(NUM_ELS, t)
        _, w2 = random_arrays(NUM_ELS, t)
        # w1 = wn.random.rand(10)
        # w2 = wn.random.rand(10)
        print(w1)
        w3 = w1 + w2
        w3 = np.exp(w3)
        w3 = w3.evaluate()

def test_cmp():
    for t in TYPES:
        n1, w1 = random_arrays(NUM_ELS, t)
        n2, w2 = random_arrays(NUM_ELS, t)

        n3 = (n1-n2)*3.0 + 400.00
        w3 = (w1-w2)*3.0 + 400.00
        # n3 = np.sin(n3)
        # w3 = np.sin(w3)
        w3 = w3.evaluate()

        assert (np.allclose(n3, w3.view(np.ndarray)))

def test_blackscholes_bug1():
    for t in TYPES:
        n1, w1 = random_arrays(NUM_ELS, t)
        n2, w2 = random_arrays(NUM_ELS, t)

        # bad.
        # w3 = w2 * np.sqrt(w1)
        # n3 = n2 * np.sqrt(n1)

        w3 = w1 * np.sqrt(w2)
        n3 = n1 * np.sqrt(n2)

        w3 = w3.evaluate()

        assert (np.allclose(n3, w3.view(np.ndarray)))

def test_boundary_error():
    for t in TYPES:
        n1, w1 = random_arrays(NUM_ELS, t)
        n2, w2 = random_arrays(NUM_ELS, t)

        n1 = n1 * n2
        w1 = w1 * w2

        n4 = n2 - n1
        w4 = w2 - w1
        w4 = w4.evaluate()
        assert (np.allclose(n4, w4.view(np.ndarray)))

def test_blackscholes_bug_commutativity():
    for t in TYPES:
        n1, w1 = random_arrays(NUM_ELS, t)
        n2, w2 = random_arrays(NUM_ELS, t)

        # n3 = n1 - n2
        # w3 = w1 - w2
        # w3 = w3.evaluate()
        # print(np.allclose(n3, w3))

        n1 = n1 * n2
        w1 = w1 * w2

        n4 = n2 - n1
        w4 = w2 - w1
        w4 = w4.evaluate()
        assert (np.allclose(n4, w4.view(np.ndarray)))

def test_dot_product():
    for t in TYPES:
        n1, w1 = random_arrays(NUM_ELS, t)
        n2, w2 = random_arrays(NUM_ELS, t)

        n3 = n1*n2
        w3 = w1*w2
        # w3 = w3.evaluate()
        n4 = np.sum(n3)
        print('n4: ', n4)

        print("before calling sum")
        w4 = np.sum(w3)
        print("sum completed in python")
        print(type(w4))
        # w4 = w4.evaluate()
        print('w4: ', w4)

        assert(np.allclose(n4, w4.view(np.ndarray)))

def test_simple_unary():
    for t in TYPES:
        n1, w1 = random_arrays(NUM_ELS, t)
        print("w1[0]: ", w1[0])

        n2 = np.exp(n1)
        w2 = np.exp(w1)
        w2 = w2.evaluate()

        assert(np.allclose(n2, w2.view(np.ndarray)))

# def test_reduction_along_axis():
    # n1 = np.random.rand(2,2)
    # w1 = wn.weldarray(n1)

    # print('n1[0]: ', n1[0])
    # w2 = np.log(w1)
    # n2 = np.log(n1)
    # n3 = np.sum(n2, axis=0)
    # print("n3: ", n3)

    # w3 = np.sum(w2, axis=0)
    # print(type(w3))
    # print(w3.shape)
    # print(w3._real_shape)
    # print(w3.weldobj.weld_code)
    # print("w3: ", w3)
    # print(np.allclose(n3, w3))

# def test_only_reduce():
    # n1, w1 = random_arrays(10, np.float64)
    # w2 = np.sum(w1)
    # print(w2)

def test_inplace_simple():
    for t in TYPES:
        n1, w1 = random_arrays(NUM_ELS, t)
        n2, w2 = random_arrays(NUM_ELS, t)
        n1 += 5.0
        w1 += 5.0
        w1 = w1.evaluate()

        assert (np.allclose(n1, w1.view(np.ndarray)))

        n1 += n2
        w1 += w2
        w1 = w1.evaluate()

        assert (np.allclose(n1, w1.view(np.ndarray)))

def test_inplace_2():
    '''
    trying to replicate failing scenario from nbody.
    '''
    for t in TYPES:
        n1, w1 = random_arrays(NUM_ELS, t)
        n2, w2 = random_arrays(NUM_ELS, t)
        n3, w3 = random_arrays(NUM_ELS, t)

        n1 += n2*n3
        w1 += w2*w3
        w1 = w1.evaluate()

        assert (np.allclose(n1, w1.view(np.ndarray)))

def test_nvvm_flag():
    '''
    temporary test.
    '''
    wn.set_nvvm(1)
    n1, w1 = random_arrays(NUM_ELS, np.float64)
    w1 = np.exp(w1)
    w1 = w1.evaluate()

test_simple()
