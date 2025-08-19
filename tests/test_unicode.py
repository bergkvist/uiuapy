from numpy.testing import assert_array_equal
import uiua

def test_unicode_find():
    find = uiua.compile('⌕')
    assert_array_equal(
        find("abracabra", "ab"),
        [1, 0, 0, 0, 0, 1, 0, 0, 0]
    )

def test_unicode_find_ab():
    find_ab = uiua.compile('⌕ "ab"')
    assert_array_equal(
        find_ab("abracabra"),
        [1, 0, 0, 0, 0, 1, 0, 0, 0]
    )
