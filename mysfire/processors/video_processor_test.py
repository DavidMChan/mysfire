import os
import torch


def test_video_processor() -> None:
    from .video_processor import VideoProcessor

    processor = VideoProcessor()
    assert processor.typestr() == "video"

    # Load the video
    video = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "example.mp4"))
    assert video is not None
    assert video["video"].shape == (3, 16, 360, 360)
    assert video["video"].max() <= 1.0
    assert video["video"].min() >= 0.0

    # Test the audio
    assert video["audio"].shape == (79872,)
    assert video["audio"].max() <= 1.0
    assert video["audio"].min() >= 0.0

    # Check for NaNs
    assert not video["video"].isnan().any()
    assert not video["audio"].isnan().any()

    # Test the collation
    batch = [
        video,
        video,
        video,
    ]
    collated = processor.collate(batch)
    assert isinstance(collated, dict)
    assert isinstance(collated["video"], torch.Tensor)
    assert isinstance(collated["audio"], torch.Tensor)

    assert collated["video"].shape == (3, 3, 16, 360, 360)
    assert collated["audio"].shape == (3, 79872)


def test_video_processor_transforms() -> None:
    from .video_processor import VideoProcessor

    processor = VideoProcessor(uniform_temporal_subsample=2, uniform_crop=120, short_side_scale=200)
    assert processor.typestr() == "video"

    # Load the video
    video = processor.load(os.path.join(os.path.dirname(__file__), "..", "..", "test", "example_data", "example.mp4"))
    assert video is not None
    assert video["video"].shape == (3, 2, 120, 120)
    assert video["video"].max() <= 1.0
    assert video["video"].min() >= 0.0

    # Test the audio
    assert video["audio"].shape == (79872,)
    assert video["audio"].max() <= 1.0
    assert video["audio"].min() >= 0.0

    # Check for NaNs
    assert not video["video"].isnan().any()
    assert not video["audio"].isnan().any()

    # Test the collation
    batch = [
        video,
        video,
        video,
    ]
    collated = processor.collate(batch)
    assert isinstance(collated, dict)
    assert isinstance(collated["video"], torch.Tensor)
    assert isinstance(collated["audio"], torch.Tensor)

    assert collated["video"].shape == (3, 3, 2, 120, 120)
    assert collated["audio"].shape == (3, 79872)
