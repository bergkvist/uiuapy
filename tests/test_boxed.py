import numpy as np
import uiua


def test_uiua_boxed_scalar_and_array():
    assert (
        repr(uiua.compile('{5 [1 2 3]}')())
        == 'array([np.uint8(5), array([1, 2, 3], dtype=uint8)], dtype=object)'
    )


def test_uiua_couple_boxed_values():
    x = np.array([1, 2, 3], dtype='float64')
    expected = f'array([array([1, 2], dtype=uint8), {x!r}], dtype=object)'
    assert repr(uiua.compile('⊟∩□1_2')(x)) == expected
