from numpy.testing import assert_array_equal
import uiua

def test_uiua_sum():
    assert uiua.compile('/+')([1, 2, 3, 4]) == 10

def test_uiua_prod():
    assert uiua.compile('/*')([1, 2, 3, 4]) == 24

def test_uiua_double():
    assert_array_equal(
        uiua.compile('*2')([1, 2, 3, 4]),
        [2, 4, 6, 8]
    )

def test_uiua_elementwise_sum():
    assert_array_equal(
        uiua.compile('+')([1, 2, 3], [4, 5, 6]),
        [5, 7, 9]
    )
