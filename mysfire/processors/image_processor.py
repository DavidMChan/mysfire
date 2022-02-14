from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch

from ._array_utils import stack_arrays_as_dict
from ._processor import Processor

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


class ImageProcessor(Processor):
    def __init__(
        self,
        resize: Optional[Union[int, Tuple[int, ...]]] = None,
        resize_interpolation: str = "bilinear",
        pad: bool = False,
    ) -> None:

        if not PIL_AVAILABLE:
            raise ImportError("PIL is not available. Please install PIL with `pip install pillow`")

        # Build the image processing transform from the keword arguments
        if TORCHVISION_AVAILABLE:
            transforms = []
            if resize is not None:
                transforms.append(torchvision.transforms.Resize(resize, interpolation=resize_interpolation))
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
        # TODO: Determine if this is the right way to deal with empty TSV fields
        if value.lower() == "none":
            return None

        with Image.open(value) as img:
            if TORCHVISION_AVAILABLE:
                return self._transform(img)
            return torch.from_numpy(np.array(img))
