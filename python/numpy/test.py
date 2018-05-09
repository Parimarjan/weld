import weldnumpy as wn
import numpy as np

NUM_ELS = 100000

def simple_test():
    w1 = wn.random.rand(10)
    w2 = wn.random.rand(10)
    print(w1)
    w3 = w1 + w2
    w3 = np.sqrt(w3)
    w3 = w3.evaluate()

def simple_cmp():
    n1 = np.random.rand(NUM_ELS)
    n2 = np.random.rand(NUM_ELS)
    w1 = wn.weldarray(n1)
    w2 = wn.weldarray(n2)

    n3 = (n1-n2)*3.0 + 400.00
    w3 = (w1-w2)*3.0 + 400.00
    # n3 = np.sin(n3)
    # w3 = np.sin(w3)
    w3 = w3.evaluate()

    print(np.allclose(n3, w3.view(np.ndarray)))

    # for i in range(len(n3)):
        # if np.abs(n3[i] - w3[i]) > 0.1:
            # wrong += 1
            # print(n3[i])
            # print(w3[i])
    # print('wrong = ', wrong)

def blackscholes_bug1():
    n1 = np.random.rand(NUM_ELS)
    n2 = np.random.rand(NUM_ELS)
    w1 = wn.weldarray(n1)
    w2 = wn.weldarray(n2)

    print("n1[0]: ", n1[0])
    print("n2[0]: ", n2[0])

    # bad.
    # w3 = w2 * np.sqrt(w1)
    # n3 = n2 * np.sqrt(n1)

    w3 = w1 * np.sqrt(w2)
    n3 = n1 * np.sqrt(n2)

    w3 = w3.evaluate()

    print(np.allclose(n3, w3.view(np.ndarray)))

def boundary_error():
    n1 = np.random.rand(NUM_ELS)
    n2 = np.random.rand(NUM_ELS)
    w1 = wn.weldarray(n1)
    w2 = wn.weldarray(n2)

    n1 = n1 * n2
    w1 = w1 * w2

    n4 = n2 - n1
    w4 = w2 - w1

    print("weld code: ", w4.weldobj.weld_code)
    w4 = w4.evaluate()
    print(np.allclose(n4, w4.view(np.ndarray)))

    print("going to do all close without casting to ndarray")
    # print(np.allclose(n4, w4))

def blackscholes_bug_commutativity():
    # not reproducible yet.
    n1 = np.random.rand(NUM_ELS)
    n2 = np.random.rand(NUM_ELS)
    w1 = wn.weldarray(n1)
    w2 = wn.weldarray(n2)

    # n3 = n1 - n2
    # w3 = w1 - w2
    # w3 = w3.evaluate()
    # print(np.allclose(n3, w3))

    n1 = n1 * n2
    w1 = w1 * w2

    n4 = n2 - n1
    w4 = w2 - w1
    w4 = w4.evaluate()
    print(np.allclose(n4, w4.view(np.ndarray)))

blackscholes_bug1()
print("blackscholes bug 1 done!")
# simple_test()
# print("simple test done!")
# simple_cmp()
# print("simple cmp done!!!!")
# blackscholes_bug_commutativity()
# print("blackscholes bug commutativity done!!!")
# boundary_error()
# print("boundary error done!!!")
