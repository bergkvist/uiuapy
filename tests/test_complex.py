from numpy.testing import assert_array_equal
import pytest
import uiua

def test_complex_scalar_sum(add):
    assert add(1 + 3j, 2 + 4j) == 3 + 7j

def test_complex_array_sum(add):
    assert_array_equal(
        add([2, 3], [1j, 2j]),
        [2 + 1j, 3 + 2j]
    )

@pytest.fixture
def add():
    return uiua.compile('+')
