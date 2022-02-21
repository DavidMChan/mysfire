import os
import torch


def test_npy_processor() -> None:
    from .npy_processor import NpyProcessor

    processor = NpyProcessor(pad=True)
    assert processor.typestr() == "npy"

    # Load the numpy tensors from npy files
    t1 = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "1.npy"))
    t2 = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "2.npy"))
    t3 = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "3.npy"))

    assert t1.shape == (3, 128)
    assert t2.shape == (7, 128)
    assert t3.shape == (1, 128)

    # Test the collation
    batch = [
        t1,
        t2,
        t3,
    ]
    collated = processor.collate(batch)
    assert isinstance(collated, dict)
    assert isinstance(collated["__root__"], torch.Tensor)
    assert isinstance(collated["seqlen"], torch.Tensor)

    assert collated["__root__"].shape == (3, 7, 128)
    assert (collated["seqlen"] == torch.Tensor([3, 7, 1])).all()


def test_npy_indexed_processor() -> None:
    from .npy_processor import NpyIndexedFileProcessor

    processor = NpyIndexedFileProcessor(
        filepath=os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "2.npy")
    )

    # Load the numpy tensors from npy files
    t1 = processor.load("0")
    t2 = processor.load("1")

    assert t1.shape == (128,)
    assert t2.shape == (128,)
