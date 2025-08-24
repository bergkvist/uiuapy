import pytest
import uiua


def test_no_args(unit):
    assert unit() == ()


def test_one_arg(unit):
    assert unit(1) == 1


def test_two_args(unit):
    assert unit(1, 2) == (1, 2)


def test_three_args(unit):
    assert unit(1, 2, 3) == (1, 2, 3)


def test_duplicate_top_of_stack():
    assert uiua.compile('.')(1, 2, 3) == (1, 1, 2, 3)


def test_unsupported_compile_flags_raise_type_error():
    with pytest.raises(TypeError):
        uiua.compile('', unsupported_flag=True)


def test_unsupported_invocation_flags_raise_type_error(unit):
    with pytest.raises(TypeError):
        unit(unsupported_flag=True)


@pytest.fixture
def unit():
    return uiua.compile('')
