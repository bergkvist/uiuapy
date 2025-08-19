import numpy as np
import pytest
import uiua


def test_float64_array_conversion(unit):
    assert repr(unit(np.array([1, 2, 3], dtype='float64'))) == 'array([1., 2., 3.])'


def test_float32_array_conversion(unit):
    assert repr(unit(np.array([1, 2, 3], dtype='float32'))) == 'array([1., 2., 3.])'


def test_float64_scalar_conversion(unit):
    assert repr(unit(np.float64(45))) == 'np.float64(45.0)'


def test_float32_scalar_conversion(unit):
    assert repr(unit(np.float32(45))) == 'np.float64(45.0)'


def test_uint64_array_conversion(unit):
    assert repr(unit(np.array([1, 2, 3], dtype='uint64'))) == 'array([1., 2., 3.])'


def test_uint32_array_conversion(unit):
    assert repr(unit(np.array([1, 2, 3], dtype='uint32'))) == 'array([1., 2., 3.])'


def test_uint16_array_conversion(unit):
    assert repr(unit(np.array([1, 2, 3], dtype='uint16'))) == 'array([1., 2., 3.])'


def test_uint64_scalar_conversion(unit):
    assert repr(unit(np.uint64(45))) == 'np.float64(45.0)'


def test_uint32_scalar_conversion(unit):
    assert repr(unit(np.uint32(45))) == 'np.float64(45.0)'


def test_uint16_scalar_conversion(unit):
    assert repr(unit(np.uint16(45))) == 'np.float64(45.0)'


def test_int64_array_conversion(unit):
    assert repr(unit(np.array([1, 2, 3], dtype='int64'))) == 'array([1., 2., 3.])'


def test_int32_array_conversion(unit):
    assert repr(unit(np.array([1, 2, 3], dtype='int32'))) == 'array([1., 2., 3.])'


def test_int16_array_conversion(unit):
    assert repr(unit(np.array([1, 2, 3], dtype='int16'))) == 'array([1., 2., 3.])'


def test_int64_scalar_conversion(unit):
    assert repr(unit(np.int64(45))) == 'np.float64(45.0)'


def test_int32_scalar_conversion(unit):
    assert repr(unit(np.int32(45))) == 'np.float64(45.0)'


def test_int16_scalar_conversion(unit):
    assert repr(unit(np.int16(45))) == 'np.float64(45.0)'


def test_uint8_array_conversion(unit):
    assert (
        repr(unit(np.array([1, 2, 3], dtype='uint8')))
        == 'array([1, 2, 3], dtype=uint8)'
    )


def test_uint8_scalar_conversion(unit):
    assert repr(unit(np.uint8(45))) == 'np.uint8(45)'


def test_bool_scalar_conversion(unit):
    assert repr(unit(True)) == 'np.uint8(1)'
    assert repr(unit(False)) == 'np.uint8(0)'


def test_complex_scalar_conversion(unit):
    assert repr(unit(5 + 13j)) == 'np.complex128(5+13j)'


def test_complex128_scalar_conversion(unit):
    assert repr(unit(np.complex128(5 + 13j))) == 'np.complex128(5+13j)'


def test_boxed_array_conversion(unit):
    jagged = np.array([np.array([1, 2, 3]), np.array([1, 2])], dtype='object')
    assert (
        repr(unit(jagged))
        == 'array([array([1., 2., 3.]), array([1., 2.])], dtype=object)'
    )


def test_unicode_conversion_array(unit):
    assert repr(unit(['hello', 'world'])) == "array(['hello', 'world'], dtype='<U5')"


def test_unicode_conversion_string(unit):
    assert repr(unit('hello')) == "np.str_('hello')"


@pytest.fixture
def unit():
    return uiua.compile('')
