import os
import torch


def test_image_processor() -> None:
    from .image_processor import ImageProcessor

    processor = ImageProcessor()
    assert processor.typestr() == "img"

    # Load the image
    image = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "teapot.jpeg"))
    assert image is not None
    assert image.shape == (3, 1380, 1840)
    assert image.max() <= 1.0
    assert image.min() >= 0.0

    # Test the collation
    batch = [
        image,
        image,
        image,
    ]
    collated = processor.collate(batch)
    assert isinstance(collated, torch.Tensor)
    assert collated.shape == (3, 3, 1380, 1840)


def test_image_processor_resize() -> None:
    from .image_processor import ImageProcessor

    processor = ImageProcessor(resize=(128, 138))
    assert processor.typestr() == "img"

    # Load the image
    image = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "teapot.jpeg"))
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 128, 138)
    assert image.max() <= 1.0
    assert image.min() >= 0.0
