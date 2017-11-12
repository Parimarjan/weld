import numpy as np
import py.test
import random
from weldnumpy import *
from test_utils import *

'''
General Notes:
    - Two multi-dim contig arrays with different strides/shapes isn't possible.
'''

'''
Views based tests.

TODO tests:
    - Scalars with multi-dim arrays.
    - somehow test that for non-contig arrays, weld's loop goes over as much contiguous elements as
      it can.
    - scalar op view.
    - a + other (view).
    - Then update the view after. This should not affect a+other's result.
'''
# ND_SHAPES = [(5,3,4), (5,4), (6,4,7,3)]
ND_SHAPES = [(5,3,4), (3,4)]

def get_noncontig_idx(shape):
    '''
    Returns indices that make an array with shape, 'shape', become a non contiguous array.

    Just gives a bunch of random multi-d indices.
    Things to test:
        - different number of idxs
        - different types of indexing styles (None, :, ... etc.)
        - different sizes, strides etc.
    '''
    # idx = []
    
    for i in range(5):
        idx = []
        for s in shape:
            # 5 tries to get non-contiguous array
            print('s: ', s)
            start = random.randint(0, s-1)
            stop = random.randint(start+1, s)
            # step = random.randint(0, 3)
            step = 1
            idx.append(slice(start, stop, step))

        idx = tuple(idx)
        a, _ = random_arrays(shape, 'float32')
        b = a[idx]
        if not b.flags.contiguous:
            break

    return idx

def test_views_non_contig_basic():
    shape = (5,5,5)
    # shape = (3,3)
    n, w = random_arrays(shape, 'float64')
    idx = get_noncontig_idx(shape)
    # idx = (slice(0, 3, 1), slice(0, 2, 1), slice(0, 3, 1))
    # idx = (slice(0, 3, 1), slice(0, 2, 1))

    n2 = n[idx]
    w2 = w[idx]
    
    # useful test to add.
    assert w2.shape == n2.shape
    assert w2.flags == n2.flags
    assert w2.strides == n2.strides
    
    print('w: ', w)
    print('n: ', n)

    print('w2: ', w2)
    print('n2: ', n2)

    
    # test unary op.
    n3 = np.sqrt(n2)
    w3 = np.sqrt(w2)
    w3 = w3.evaluate()
    
    # print(w3.shape)
    # print('n3.shape: ', n3.shape)
    print('w3 = ', w3)
    print('n3 = ', n3)

    # assert np.allclose(n, w)
    # assert np.allclose(n3, w3)
    # for i in range(n3.shape[0]):
        # for j in range(n3.shape[1]):
            # assert n3[i][j] == w3[i][j], 'test'

    # test binary op.

# def test_views_non_contig_inplace_unary():
    # for shape in ND_SHAPES:
        # n, w = random_arrays(shape, 'float32')
        # idx = get_noncontig_idx(shape)

        # n2 = n[idx]
        # w2 = w[idx]

        # # unary op test.
        # n2 = np.sqrt(n2, out=n2)
        # w2 = np.sqrt(w2, out=w2)

        # assert np.allclose(n2, w2)
        # assert np.allclose(n, w)

def test_views_non_contig_newarray_binary():
    '''
    binary involves a few cases that we need to test:
        - non-contig + contig
        - contig + non-contig
        - non-contig + non-contig
    '''
    ND_SHAPES = [(5,5)] 
    BINARY_OPS = [np.add]

    for shape in ND_SHAPES:
        n, w = random_arrays(shape, 'float32')
        n2, w2 = random_arrays(shape, 'float32')
        idx = get_noncontig_idx(shape)

        nv1 = n[idx]
        wv1 = w[idx]
        nv2 = n2[idx]
        wv2 = w2[idx]

        for op in BINARY_OPS:
            print('op = ', op)
            nv3 = op(nv1, nv2)
            wv3 = op(wv1, wv2)
            wv3 = wv3.evaluate()

            assert nv3.shape == wv3.shape, 'shape not same'

            assert np.allclose(nv2, wv2)
            assert np.allclose(nv1, wv1)
            assert np.allclose(nv3, wv3)

# def test_views_non_contig_inplace_binary1():
    # '''
    # TODO: separate out case with updated other into new test.

    # binary involves a few cases that we need to test:
    # In place ops, consider the case:
        # - non-contig + non-contig

    # Note: these don't test for performance, for instance, if the non-contig array is being
    # evaluated with the max contiguity etc.
    # '''
    # for shape in ND_SHAPES:
        # n, w = random_arrays(shape, 'float32')
        # n2, w2 = random_arrays(shape, 'float32')
        # idx = get_noncontig_idx(shape)
        # print('idx : ', idx)

        # nv1 = n[idx]
        # wv1 = w[idx]
        # nv2 = n2[idx]
        # wv2 = w2[idx]

        # # update other from before.
        # # important: we need to make sure the case with the second array having some operations
        # # stored in it is dealt with.
        # wv2 = np.sqrt(wv2, out=wv2)
        # nv2 = np.sqrt(nv2, out=nv2)

        # for op in BINARY_OPS:
            # print('op = ', op)
            # nv1 = op(nv1, nv2, out=nv1)
            # wv1 = op(wv1, wv2, out=wv1)

            # # when we evaluate a weldarray_view, the view properties (parent array etc) must be preserved in the
            # # returned array.
            # wv1 = wv1.evaluate()

            # # print(wv1.flags)
            # # print(wv1._weldarray_weldview)
            # # print('nv1: ', nv1)
            # # print('wv1: ', wv1)

            # assert np.allclose(nv2, wv2)
            # assert np.allclose(nv1, wv1)
            # assert np.allclose(n, w)
            # assert np.allclose(n2, w2)

            # print('***********ENDED op {} ************'.format(op))
        # print('***********ENDED shape {} ************'.format(shape))

# def test_views_non_contig_inplace_binary2():
    # '''
    # binary involves a few cases that we need to test:
    # In place ops:
        # - contig + non-contig
        # - non-contig + contig
            # - surprisingly these two cases aren't so trivial because we can't just loop over the
              # contiguous array as there would be no clear way to map the indices to the non-contig
              # case.
            # - it might even be better sometimes for performance to convert the non-contig case to
              # contig?

    # These cases are similar to the newarray cases though and both these tests should pass together
    # based on current implementation.
    # TODO: write the test.
    # '''
    # for shape in ND_SHAPES:
        # n, w = random_arrays(shape, 'float32')
        # n2, w2 = random_arrays(shape, 'float32')
        # idx = get_noncontig_idx(shape)
        # # TODO: write test.
        # pass

# def test_views_non_contig_inplace_other_updates():
    # '''
    # '''
    # for shape in ND_SHAPES:
        # n, w = random_arrays(shape, 'float32')
        # n2, w2 = random_arrays(shape, 'float32')
        # idx = get_noncontig_idx(shape)

        # nv1 = n[idx]
        # wv1 = w[idx]
        # nv2 = n2[idx]
        # wv2 = w2[idx]

        # # update other from before.
        # # important: we need to make sure the case with the second array having some operations
        # # stored in it is dealt with.
        # wv2 = np.sqrt(wv2, out=wv2)
        # nv2 = np.sqrt(nv2, out=nv2)
        # # wv2 = wv2.evaluate()

        # op = np.subtract
        # nv1 = op(nv1, nv2, out=nv1)
        # wv1 = op(wv1, wv2, out=wv1)

        # nv2 = np.log(nv2, out=nv2)
        # wv2 = np.log(wv2, out=wv2)
        # wv2 = wv2.evaluate()

        # n2 = np.sqrt(n2, out=n2)
        # w2 = np.sqrt(w2, out=w2)

        # # when we evaluate a weldarray_view, the view properties (parent array etc) must be preserved in the
        # # returned array.
        # wv1 = wv1.evaluate()

        # assert np.allclose(nv2, wv2)
        # assert np.allclose(nv1, wv1)
        # assert np.allclose(n, w)
        # assert np.allclose(n2, w2)

        # print('***********ENDED shape {} ************'.format(shape))


# def test_views_non_contig_inplace_binary_mess():
    # '''
    # Just mixes a bunch of operations on top of the previous setup.
    # '''
    # for shape in ND_SHAPES:
        # n, w = random_arrays(shape, 'float32')
        # n2, w2 = random_arrays(shape, 'float32')
        # idx = get_noncontig_idx(shape)
        # nv1 = n[idx]
        # wv1 = w[idx]
        # nv2 = n2[idx]
        # wv2 = w2[idx]
        # # TODO: write test.

'''
General Tests.
'''
def test_reshape():
    n, w = random_arrays(36, 'float32')
    n = n.reshape((6,6))
    w = w.reshape((6,6))

    assert isinstance(w, weldarray)
    assert w.shape == n.shape
    assert w.strides == n.strides

'''
TODO: Set up multi-dimensional array creation routines.

    - create new arrays in different ways: transpose, reshape, concatenation, horizontal/vertical etc.
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
More advanced tests.
'''

'''
Broadcasting based tests.
'''
# test_views_non_contig_basic()
# test_views_non_contig_newarray_binary()
