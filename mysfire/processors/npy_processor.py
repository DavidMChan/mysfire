from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch

from ._array_utils import stack_arrays_as_dict
from ._processor import S3Processor


class NpyProcessor(S3Processor):
    def __init__(
        self,
        pad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self._pad = pad

    @classmethod
    def typestr(cls) -> str:
        return "npy"

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
        with self.resolve_to_local(value) as f:
            return torch.from_numpy(np.load(f))  # type: ignore


class NpyIndexedFileProcessor(S3Processor):
    def __init__(
        self,
        filepath: str,
        pad: bool = False,
        **kwargs: Any,
    ) -> None:

        super().__init__(**kwargs)

        with self.resolve_to_local(filepath) as f:
            self._data = np.load(f)  # type: ignore

        self._pad = pad

    @classmethod
    def typestr(cls) -> str:
        return "npy.indexed_file"

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
        return torch.from_numpy(self._data[int(value)]) if value else None
