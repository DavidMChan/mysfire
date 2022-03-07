from typing import Dict, List, Optional, Union, Tuple, Any

import torch
import pathlib

from ._array_utils import stack_arrays_as_dict
from ._processor import S3Processor
from . import register_processor

PYTORCH_VIDEO_AVAILABLE = False
try:
    from pytorchvideo.transforms import ShortSideScale, UniformTemporalSubsample
    from pytorchvideo.transforms.functional import uniform_crop as uniform_crop_fn

    PYTORCH_VIDEO_AVAILABLE = True
except ImportError:
    pass

TORCHVISION_AVAILABLE = False
try:
    from torchvision.transforms import Compose, Lambda, Normalize

    TORCHVISION_AVAILABLE = True
except ImportError:
    pass

PYAV_AVAILABLE = False
try:
    import av

    PYAV_AVAILABLE = True
except ImportError:
    pass

# TORCHAUDIO_AVAILABLE = False
# try:
#     from torchaudio.transforms import Resample

#     TORCHAUDIO_AVAILABLE = True
# except ImportError:
#     pass


def _decode_av(input_: av.container.Container) -> Tuple[torch.Tensor, torch.Tensor]:
    # Set up pyav for fast decoding
    input_.streams.video[0].thread_type = "AUTO"
    input_.streams.audio[0].thread_type = "AUTO"

    # Allocate memory for the video/audio
    _video = torch.empty(
        input_.streams.video[0].frames,
        input_.streams.video[0].height,
        input_.streams.video[0].width,
        3,
        dtype=torch.uint8,
    )
    _audio = torch.empty(
        input_.streams.audio[0].frames, 1024, dtype=torch.float32
    )  # It remains to be seen if this is set at 1024

    audio_idx, video_idx = 0, 0
    for frame in input_.decode(video=0, audio=0):
        if isinstance(frame, av.audio.frame.AudioFrame):
            base_frame = torch.from_numpy(frame.to_ndarray()).mean(dim=0)  # Mix down audio to mono
            _audio[audio_idx] = torch.nn.functional.pad(base_frame, (0, 1024 - base_frame.shape[0]))
            audio_idx += 1
        else:
            _video[video_idx] = torch.from_numpy(frame.to_ndarray(format="rgb24"))
            video_idx += 1

    return _video.permute(3, 0, 1, 2), _audio.reshape(-1).clip(0, 1)


def _decode_v(input_: av.container.Container) -> torch.Tensor:
    input_.streams.video[0].thread_type = "AUTO"
    _video = torch.empty(
        input_.streams.video[0].frames,
        input_.streams.video[0].height,
        input_.streams.video[0].width,
        3,
        dtype=torch.uint8,
    )
    for idx, frame in enumerate(input_.decode(video=0)):
        _video[idx] = torch.from_numpy(frame.to_ndarray(format="rgb24"))

    return _video.permute(3, 0, 1, 2)


def _decode_a(input_: av.container.Container) -> torch.Tensor:
    input_.streams.audio[0].thread_type = "AUTO"
    _audio = torch.empty(
        input_.streams.audio[0].frames, 1024, dtype=torch.float32
    )  # It remains to be seen if this is set at 1024
    for idx, frame in enumerate(input_.decode(audio=0)):
        base_frame = torch.from_numpy(frame.to_ndarray()).mean(dim=0)  # Mix down audio to mono
        _audio[idx] = torch.nn.functional.pad(base_frame, (0, 1024 - base_frame.shape[0]))
    return _audio.reshape(-1).clip(0, 1)


def load_mp4_video(file_path: Union[pathlib.Path, str]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    # Open and read the video/audio file
    input_ = av.open(str(file_path), "r")
    video, audio = None, None
    try:
        if len(input_.streams.audio) > 0 and len(input_.streams.video) > 0:
            video, audio = _decode_av(input_)
        elif len(input_.streams.video) > 0:
            video = _decode_v(input_)
        elif len(input_.streams.audio) > 0:
            audio = _decode_a(input_)
    except Exception as ex:
        print(ex)
    finally:
        input_.close()

    return video, audio


@register_processor
class VideoProcessor(S3Processor):
    def __init__(
        self,
        uniform_temporal_subsample: Optional[int] = None,
        uniform_crop: Optional[int] = None,
        short_side_scale: Optional[int] = None,
        pad: bool = False,
        **kwargs: Any,
    ):

        # Guards for optional dependencies
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is not available. Please install torchvision with `pip install torchvision`")

        # if not TORCHAUDIO_AVAILABLE:
        #     raise ImportError("torchaudio is not available. Please install torchaudio with `pip install torchaudio`")
        super().__init__(**kwargs)

        self._uniform_temporal_subsample = uniform_temporal_subsample
        self._uniform_crop = uniform_crop
        self._short_side_scale = short_side_scale
        self._pad = pad

        video_transforms = []
        if self._uniform_temporal_subsample is not None:
            video_transforms.append(UniformTemporalSubsample(self._uniform_temporal_subsample))
        if self._short_side_scale is not None:
            video_transforms.extend(
                (
                    Lambda(lambda x: x.float()),
                    ShortSideScale(self._short_side_scale),
                )
            )

        if self._uniform_crop is not None:
            video_transforms.append(Lambda(lambda x: uniform_crop_fn(x, self._uniform_crop, 1)))
        video_transforms.append(Lambda(lambda x: x / 255.0))  # Always normalize the video

        self._video_transform = Compose(video_transforms)
        self._audio_transform = Compose([])

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
        with self.resolve_to_local(value) as local_path:
            video, audio = load_mp4_video(local_path)

        frames = self._video_transform(video) if video is not None else None
        audio = self._audio_transform(audio) if audio is not None else None

        return {
            "video": frames,
            "audio": audio,
        }


@register_processor
class FixedSizeOutputVideoProcessor(S3Processor):
    def __init__(
        self,
        video_shape: Tuple[int, int, int, int],
        audio_shape: Tuple[int, int],
        short_side_scale: Optional[int] = None,
        **kwargs: Any,
    ):
        """Specifies a fixed output shape for the video. If the video (or audio) doesn't conform to this output shape,
        it is padded with zeros.

        Limitations:
            - Only works with the same height/width
            - Only works with 3 RGB channels
            - Only works with 1 mono audio channel

        Args:
            video_shape (Tuple[int]): The shape of the video to produce as a tuple: (Channels, Time, Height, Width)
            audio_shape (Tuple[int]): The shape of the audio to produce as a tuple: (Time, Channels)
        """

        # Guards for optional dependencies
        if not PYAV_AVAILABLE:
            raise ImportError("py-av is not available. Please install py-av with `pip install av`")
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is not available. Please install torchvision with `pip install torchvision`")

        super().__init__(**kwargs)

        # if not TORCHAUDIO_AVAILABLE:
        #     raise ImportError("torchaudio is not available. Please install torchaudio with `pip install torchaudio`")

        assert len(video_shape) == 4, "Video shape must be a tuple of length 4"
        assert len(audio_shape) == 2, "Audio shape must be a tuple of length 2"
        assert video_shape[2] == video_shape[3], "Video height and width must be equal"
        assert audio_shape[1] == 1, "Audio channels must be 1"
        assert video_shape[0] == 3, "Video must have 3 channels"

        self._video_shape = video_shape
        self._audio_shape = audio_shape

        self._uniform_temporal_subsample = video_shape[1]
        self._uniform_crop = video_shape[2]
        self._short_side_scale = short_side_scale

        video_transforms = [
            UniformTemporalSubsample(self._uniform_temporal_subsample),
            Lambda(lambda x: x.float()),
            ShortSideScale(self._short_side_scale) if self._short_side_scale else Lambda(lambda x: x),
            Lambda(lambda x: uniform_crop_fn(x, self._uniform_crop, 1)),
            Lambda(lambda x: x / 255.0),  # Always normalize the video
            Lambda(lambda x: x.permute(1, 0, 2, 3)),
            Normalize(
                mean=(0.43216, 0.394666, 0.37645),
                std=(0.22803, 0.22145, 0.216989),
            ),
            Lambda(lambda x: x.permute(1, 0, 2, 3)),
        ]

        self._video_transform = Compose(video_transforms)
        self._audio_transform = Compose([])

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
            "frames": torch.stack([b["frames"] for b in batch], dim=0),
            "audio": torch.stack([b["audio"] for b in batch], dim=0),
            "frames_sequence_mask": torch.stack([b["frames_sequence_mask"] for b in batch], dim=0),
            "audio_sequence_mask": torch.stack([b["audio_sequence_mask"] for b in batch], dim=0),
        }

    def __call__(self, value: str) -> Dict[str, torch.Tensor]:
        # Load the video
        with self.resolve_to_local(value) as local_path:
            video, audio = load_mp4_video(local_path)

        frames = self._video_transform(video) if video is not None else torch.zeros(self._video_shape)
        audio_data = self._audio_transform(audio) if audio is not None else torch.zeros(self._audio_shape)

        # Pad the audio to match the expected fixed length
        if len(audio_data) > self._audio_shape[0]:
            audio_data = audio_data[: self._audio_shape[0]]
        elif len(audio_data) < self._audio_shape[0]:
            audio_data = torch.cat([audio_data, torch.zeros(self._audio_shape[0] - len(audio_data))])

        # Build the sequence masks
        video_sequence_mask = (
            torch.ones(self._video_shape[1], dtype=torch.bool)
            if video is not None
            else torch.zeros(frames.shape[1], dtype=torch.bool)
        )
        audio_sequence_mask = (
            torch.ones(self._audio_shape[0], dtype=torch.bool)
            if audio is not None
            else torch.zeros(audio_data.shape[0], dtype=torch.bool)
        )

        # Unstack the audio to a single channel
        audio_data = audio_data.view(self._audio_shape[0], -1)

        # Filter NaNs in audio data
        audio_data = torch.nan_to_num(audio_data)

        assert self._video_shape == frames.shape, "Internal error: Video shape must be {} (It's {})".format(
            self._video_shape, frames.shape
        )
        assert self._audio_shape == audio_data.shape, "Internal Error: Audio shape must be {}  (It's {})".format(
            self._audio_shape, audio_data.shape
        )

        return {
            "frames": frames,
            "audio": audio_data,
            "frames_sequence_mask": video_sequence_mask,
            "audio_sequence_mask": audio_sequence_mask,
        }
