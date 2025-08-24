import numpy as np
import uiua
from numpy.testing import assert_array_equal


def test_uiua_sum():
    assert uiua.compile('/+')([512, 512, 1024, 2048]) == 4096


def test_uiua_prod():
    assert uiua.compile('/*')([-4, 3, 2, 1]) == -24


def test_uiua_double():
    assert_array_equal(uiua.compile('*2')([1, -2, 3, -4]), [2, -4, 6, -8])


def test_uiua_elementwise_sum():
    assert_array_equal(uiua.compile('+')([1, 2, 3], [4, 5, 6]), [5, 7, 9])


def test_uiua_sub():
    assert uiua.compile('-')(13, 7) == -6


def test_2d_array():
    assert_array_equal(
        uiua.compile('°△3_4')(), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    )


def test_array_sum_with_threads():
    xs = np.linspace(0, 1, 100_000)
    regular_sum = uiua.compile('/+')
    threaded_sum = uiua.compile('/+', allow_threads=True)
    assert threaded_sum(xs) == 50_000
    assert regular_sum(xs, allow_threads=True) == 50_000
    assert threaded_sum(xs, allow_threads=False) == 50_000
