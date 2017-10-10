# import numpy as np
import weldnumpy as wnp
import py.test
import random

def test_zeros():
    a = wnp.zeros(10)
    assert isinstance(a, wnp.weldarray)

def test_random():
    a = wnp.random.rand(10)
    assert isinstance(a, wnp.weldarray)
    
    b = wnp.random.randint(10)
    assert isinstance(a, wnp.weldarray)

    c = wnp.random.choice(a, 5)
    # assert isinstance(c, wnp.weldarray)



