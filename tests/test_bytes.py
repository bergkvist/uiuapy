import numpy as np
import uiua

def test_eq3():
    eq3 = uiua.compile('=3')
    assert repr(eq3(np.array([1, 2, 3, 4], dtype='uint8'))) == 'array([0, 0, 1, 0], dtype=uint8)'
