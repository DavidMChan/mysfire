import pytest
import torch


def test_int_processor() -> None:
    from .base_types_processor import IntProcessor

    processor = IntProcessor()
    assert processor.typestr() == "int"
    assert (processor.collate([1, 2, 3]) == torch.tensor([1, 2, 3])).all()
    assert processor.__call__("1") == 1
    assert processor.__call__("") is None

    with pytest.raises(ValueError):
        processor.__call__("a")


def test_float_processor() -> None:
    from .base_types_processor import FloatProcessor

    processor = FloatProcessor()
    assert processor.typestr() == "float"
    assert (processor.collate([1.0, 2.0, 3.0]) == torch.tensor([1.0, 2.0, 3.0])).all()
    assert processor.__call__("1.0") == 1.0
    assert processor.__call__("") is None

    with pytest.raises(ValueError):
        processor.__call__("a")


def test_string_processor() -> None:
    from .base_types_processor import StringProcessor

    processor = StringProcessor()
    assert processor.typestr() == "str"
    assert processor.collate(["a", "b", "c"]) == ["a", "b", "c"]
    assert processor.__call__("a") == "a"
    assert processor.__call__("") is None


def test_string_list_processor() -> None:
    from .base_types_processor import StringListProcessor

    processor = StringListProcessor()
    assert processor.typestr() == "str.list"
    assert processor.collate([["a", "b", "c"], ["d", "e", "f"]]) == [["a", "b", "c"], ["d", "e", "f"]]
    assert processor.__call__("a###b###c") == ["a", "b", "c"]
    assert processor.__call__("") is None
