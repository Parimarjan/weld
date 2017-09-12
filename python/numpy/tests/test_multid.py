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
    - somehow test that for non-contig arrays, weld's loop goes over as much contiguous elements as
      it can.
    - scalar op view.
'''

def get_idxs(shape=None):
    '''
    TODO: Need to generate idxs based on the shape parameter...

    Just gives a bunch of random multi-d indices.
    Things to test:
        - different number of idxs
        - different types of indexing styles (None, :, ... etc.)
        - different sizes, strides etc.
    '''
    idxs = []

    idxs.append((slice(2,4,1),slice(1,3,1), slice(0,1,1)))
    # For future, adding None and stuff.
    # (slice(None,None,2),slice(3,None,None), slice(1,None,3))

    return idxs

def test_views_non_contig_basic():
    n, w = random_arrays((5,5,5), 'float32')
    idxs = get_idxs()

    for idx in idxs:
        n2 = n[idx]
        w2 = w[idx]

        # useful test to add.
        assert w2.shape == n2.shape
        assert w2.flags == n2.flags
        assert w2.strides == n2.strides

        # test unary op.
        n3 = np.log(n2)
        w3 = np.log(w2)
        w3 = w3.evaluate()

        # test binary op.

        assert np.allclose(n3, w3)
        assert np.allclose(n, w)

def test_views_non_contig_inplace_unary():
    n, w = random_arrays((5,5,5), 'float32')
    idxs = get_idxs()

    for idx in idxs:
        n2 = n[idx]
        w2 = w[idx]

        # unary op test.
        n2 = np.sqrt(n2, out=n2)
        w2 = np.sqrt(w2, out=w2)

        assert np.allclose(n2, w2)
        assert np.allclose(n, w)

def test_views_non_contig_newarray_binary():
    '''
    binary involves a few cases that we need to test:
        - non-contig + contig
        - contig + non-contig
        - non-contig + non-contig
    '''
    n, w = random_arrays((5,5,5), 'float32')
    n2, w2 = random_arrays((5,5,5), 'float32')
    idxs = get_idxs()

    for idx in idxs:
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

def test_views_non_contig_inplace_binary1():
    '''
    TODO: separate out case with updated other into new test.

    binary involves a few cases that we need to test:
    In place ops, consider the case:
        - non-contig + non-contig

    Note: these don't test for performance, for instance, if the non-contig array is being
    evaluated with the max contiguity etc.
    '''
    n, w = random_arrays((5,5,5), 'float32')
    n2, w2 = random_arrays((5,5,5), 'float32')
    idxs = get_idxs()

    for idx in idxs:
        nv1 = n[idx]
        wv1 = w[idx]
        nv2 = n2[idx]
        wv2 = w2[idx]

        # update other from before.
        # important: we need to make sure the case with the second array having some operations
        # stored in it is dealt with.
        wv2 = np.sqrt(wv2, out=wv2)
        nv2 = np.sqrt(nv2, out=nv2)

        for op in BINARY_OPS:
            print('op = ', op)
            nv1 = op(nv1, nv2, out=nv1)
            wv1 = op(wv1, wv2, out=wv1)

            # when we evaluate a weldarray_view, the view properties (parent array etc) must be preserved in the
            # returned array.
            wv1 = wv1.evaluate()

            # print(wv1.flags)
            # print(wv1._weldarray_weldview)
            # print('nv1: ', nv1)
            # print('wv1: ', wv1)

            assert np.allclose(nv2, wv2)
            assert np.allclose(nv1, wv1)
            assert np.allclose(n, w)
            assert np.allclose(n2, w2)

            print('***********ENDED op {} ************'.format(op))

def test_views_non_contig_inplace_binary2():
    '''
    binary involves a few cases that we need to test:
    In place ops:
        - contig + non-contig
        - non-contig + contig
            - surprisingly these two cases aren't so trivial because we can't just loop over the
              contiguous array as there would be no clear way to map the indices to the non-contig
              case.
            - it might even be better sometimes for performance to convert the non-contig case to
              contig?

    These cases are similar to the newarray cases though and both these tests should pass together
    based on current implementation.
    TODO: write the test.
    '''
    n, w = random_arrays((5,5,5), 'float32')
    n2, w2 = random_arrays((5,5,5), 'float32')
    idxs = get_idxs()

    for idx in idxs:
        pass

def test_views_non_contig_inplace_binary_mess():
    '''
    Just mixes a bunch of operations on top of the previous setup.
    '''
    n, w = random_arrays((5,5,5), 'float32')
    n2, w2 = random_arrays((5,5,5), 'float32')
    idxs = get_idxs()

    for idx in idxs:
        nv1 = n[idx]
        wv1 = w[idx]
        nv2 = n2[idx]
        wv2 = w2[idx]

        # TODO: Create some messy ops here.

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
