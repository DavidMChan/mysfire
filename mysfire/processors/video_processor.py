from typing import Dict, List, Optional, Union, Tuple

import torch

from ._array_utils import stack_arrays_as_dict
from ._processor import Processor

PYTORCH_VIDEO_AVAILABLE = False
try:
    from pytorchvideo.data.encoded_video import EncodedVideo
    from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
    from pytorchvideo.transforms.functional import uniform_crop as uniform_crop_fn

    PYTORCH_VIDEO_AVAILABLE = True
except ImportError:
    pass

TORCHVISION_AVAILABLE = False
try:
    from torchvision.transforms import Compose, Lambda

    TORCHVISION_AVAILABLE = True
except ImportError:
    pass

# TORCHAUDIO_AVAILABLE = False
# try:
#     from torchaudio.transforms import Resample

#     TORCHAUDIO_AVAILABLE = True
# except ImportError:
#     pass


class VideoProcessor(Processor):
    def __init__(
        self,
        uniform_temporal_subsample: Optional[str] = None,
        uniform_crop: Optional[str] = None,
        short_side_scale: Optional[str] = None,
        pad: str = "false",
    ):

        # Guards for optional dependencies
        if not PYTORCH_VIDEO_AVAILABLE:
            raise ImportError(
                "pytorchvideo is not available. Please install pytorchvideo with `pip install pytorchvideo`"
            )
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is not available. Please install torchvision with `pip install torchvision`")

        # if not TORCHAUDIO_AVAILABLE:
        #     raise ImportError("torchaudio is not available. Please install torchaudio with `pip install torchaudio`")

        self._uniform_temporal_subsample = int(uniform_temporal_subsample) if uniform_temporal_subsample else None
        self._uniform_crop = int(uniform_crop) if uniform_crop else None
        self._short_side_scale = int(short_side_scale) if short_side_scale else None
        self._pad = pad.lower() in ("yes", "true", "t", "1")

        video_transforms = []
        if self._uniform_temporal_subsample is not None:
            video_transforms.append(UniformTemporalSubsample(self._uniform_temporal_subsample))
        if self._short_side_scale is not None:
            video_transforms.append(ShortSideScale(self._short_side_scale))
        if self._uniform_crop is not None:
            video_transforms.append(Lambda(lambda x: uniform_crop_fn(x, self._uniform_crop, 1)))
        video_transforms.append(Lambda(lambda x: x / 255.0))  # Always normalize the video

        self._video_transform = ApplyTransformToKey(key="video", transform=Compose(video_transforms))
        self._audio_transform = ApplyTransformToKey(key="audio", transform=Compose([]))

    @classmethod
    def typestr(cls) -> str:
        return "video"

    def collate(
        self, batch: List[Dict[str, Optional[torch.Tensor]]]
    ) -> Dict[
        str,
        Optional[
            Union[
                torch.Tensor,
                Dict[str, Union[Optional[torch.Tensor], Optional[List[Optional[torch.Tensor]]]]],
                List[Optional[torch.Tensor]],
            ]
        ],
    ]:
        return {
            "video": stack_arrays_as_dict([b["video"] for b in batch], pad=self._pad),
            "audio": stack_arrays_as_dict([b["audio"] for b in batch], pad=self._pad),
        }

    def __call__(self, value: str) -> Dict[str, Optional[torch.Tensor]]:
        # Load the video
        video = EncodedVideo.from_path(value, decode_audio=True)
        video_data = video.get_clip(0, video.duration)
        frames = self._video_transform(video_data)["video"]

        # Load the audio
        audio_data = None
        if "audio" in video_data and video_data["audio"] is not None:
            audio_data = self._audio_transform(video_data)["audio"]

        return {
            "video": frames,
            "audio": audio_data,
        }


class FixedSizeOutputVideoProcessor(Processor):
    def __init__(
        self,
        video_shape: Tuple[int, int, int, int],
        audio_shape: Tuple[int, int],
        short_side_scale: Optional[int] = None,
    ):
        """Specifies a fixed output shape for the video. If the video (or audio) doesn't conform to this output shape,
        it is padded with zeros.

        Limitations:
            - Only works with the same height/width
            - Only works with 3 RGB channels
            - Only works with 1 mono audio channel

        Args:
            video_shape (Tuple[int]): The shape of the video to produce as a tuple: (Time, Height, Width, Channels)
            audio_shape (Tuple[int]): The shape of the audio to produce as a tuple: (Time, Channels)
        """

        # Guards for optional dependencies
        if not PYTORCH_VIDEO_AVAILABLE:
            raise ImportError(
                "pytorchvideo is not available. Please install pytorchvideo with `pip install pytorchvideo`"
            )
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is not available. Please install torchvision with `pip install torchvision`")

        # if not TORCHAUDIO_AVAILABLE:
        #     raise ImportError("torchaudio is not available. Please install torchaudio with `pip install torchaudio`")

        assert len(video_shape) == 4, "Video shape must be a tuple of length 4"
        assert len(audio_shape) == 2, "Audio shape must be a tuple of length 2"
        assert video_shape[1] == video_shape[2], "Video height and width must be equal"
        assert audio_shape[1] == 1, "Audio channels must be 1"
        assert video_shape[-1] == 3, "Video must have 3 channels"

        self._video_shape = video_shape
        self._audio_shape = audio_shape

        self._uniform_temporal_subsample = video_shape[0]
        self._uniform_crop = video_shape[1]
        self._short_side_scale = short_side_scale

        video_transforms = [
            UniformTemporalSubsample(self._uniform_temporal_subsample),
            ShortSideScale(self._short_side_scale) if self._short_side_scale else Lambda(lambda x: x),
            Lambda(lambda x: uniform_crop_fn(x, self._uniform_crop, 1)),
            Lambda(lambda x: x / 255.0),  # Always normalize the video
        ]

        self._video_transform = ApplyTransformToKey(key="video", transform=Compose(video_transforms))
        self._audio_transform = ApplyTransformToKey(key="audio", transform=Compose([]))

    @classmethod
    def typestr(cls) -> str:
        return "video.fixed_size"

    def collate(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[
        str,
        Optional[
            Union[
                torch.Tensor,
                Dict[str, Union[Optional[torch.Tensor], Optional[List[Optional[torch.Tensor]]]]],
                List[Optional[torch.Tensor]],
            ]
        ],
    ]:
        return {
            "video": torch.stack([b["video"] for b in batch], dim=0),
            "audio": torch.stack([b["audio"] for b in batch], dim=0),
            "video_sequence_mask": torch.stack([b["video_sequence_mask"] for b in batch], dim=0),
            "audio_sequence_mask": torch.stack([b["audio_sequence_mask"] for b in batch], dim=0),
        }

    def __call__(self, value: str) -> Dict[str, torch.Tensor]:
        # Load the video
        video = EncodedVideo.from_path(value, decode_audio=True)
        video_data = video.get_clip(0, video.duration)
        frames = self._video_transform(video_data)["video"]

        # Load the audio
        audio_data = None
        if "audio" in video_data and video_data["audio"] is not None:
            audio_data = self._audio_transform(video_data)["audio"]

        # Build the sequence masks
        video_sequence_mask = torch.ones(self._video_shape[0], dtype=torch.bool)
        audio_sequence_mask = torch.ones(self._audio_shape[0], dtype=torch.bool)

        if frames is None:
            frames = torch.zeros(*self._video_shape)
            video_sequence_mask = torch.zeros(frames.shape[0], dtype=torch.bool)
        if audio_data is None:
            audio_data = torch.zeros(*self._audio_shape)
            audio_sequence_mask = torch.zeros(audio_data.shape[0], dtype=torch.bool)

        assert self._video_shape == frames.shape, "Internal error: Video shape must be {}".format(self._video_shape)
        assert self._audio_shape == audio_data.shape, "Internal Error: Audio shape must be {}".format(self._audio_shape)

        return {
            "video": frames,
            "audio": audio_data,
            "video_sequence_mask": video_sequence_mask,
            "audio_sequence_mask": audio_sequence_mask,
        }
