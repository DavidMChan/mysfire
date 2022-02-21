import os
import torch


def test_pytorch_processor() -> None:
    from .pytorch_processor import PtProcessor

    processor = PtProcessor(pad=True)
    assert processor.typestr() == "pt"

    # Load the pytorch tensors from pt files
    t1 = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "1.pt"))
    t2 = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "2.pt"))
    t3 = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "3.pt"))

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
