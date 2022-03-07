from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch

from ._array_utils import stack_arrays_as_dict
from ._processor import S3Processor
from . import register_processor

TORCHVISION_AVAILABLE = False
try:
    import torchvision

    TORCHVISION_AVAILABLE = True
except ImportError:
    pass

PIL_AVAILABLE = False
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    pass


@register_processor
class ImageProcessor(S3Processor):
    def __init__(
        self,
        resize: Optional[Union[int, Tuple[int, ...]]] = None,
        resize_interpolation: str = "bilinear",
        pad: bool = False,
        **kwargs: Any,
    ) -> None:

        if not PIL_AVAILABLE:
            raise ImportError("PIL is not available. Please install PIL with `pip install pillow`")

        super().__init__(**kwargs)

        # Build the image processing transform from the keword arguments
        if TORCHVISION_AVAILABLE:
            transforms = []
            if resize is not None:
                rim = torchvision.transforms.functional.InterpolationMode(resize_interpolation)
                transforms.append(torchvision.transforms.Resize(resize, interpolation=rim))
            transforms.append(torchvision.transforms.ToTensor())
            self._transform = torchvision.transforms.Compose(transforms)
        elif resize is not None:
            raise ImportError(
                "Torchvision is not available, but transforms are requested."
                " Please install torchvision with `pip install torchvision`"
            )

        self._pad = pad

    @classmethod
    def typestr(cls) -> str:
        return "img"

    def collate(
        self, batch: List[Optional[torch.Tensor]]
    ) -> Optional[
        Union[
            torch.Tensor,
            Dict[str, Union[Optional[torch.Tensor], Optional[List[Optional[torch.Tensor]]]]],
            List[Optional[torch.Tensor]],
        ]
    ]:
        return stack_arrays_as_dict(batch, self._pad)

    def __call__(self, value: str) -> Optional[torch.Tensor]:
        with self.resolve_to_local(value) as fp:
            with Image.open(fp) as img:
                if TORCHVISION_AVAILABLE:
                    return self._transform(img)
                return torch.from_numpy(np.array(img))
