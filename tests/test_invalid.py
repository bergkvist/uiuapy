import pytest
import uiua

def test_dict_not_allowed():
    unit = uiua.compile('')
    with pytest.raises(ValueError):
        unit({})
