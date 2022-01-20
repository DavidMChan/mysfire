from typing import Dict, List, Optional, Union

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
    def typestr(cls):
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
