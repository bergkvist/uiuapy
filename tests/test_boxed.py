import uiua

def test_uiua_boxed_scalar_and_array():
    assert repr(uiua.compile("{5 [1 2 3]}")()) == "array([np.uint8(5), array([1, 2, 3], dtype=uint8)], dtype=object)"
