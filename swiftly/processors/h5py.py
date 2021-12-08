from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ._array_utils import stack_arrays_as_dict
from ._processor import Processor

H5PY_AVAILABLE = False
try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    pass


class H5PyDatasetProcessor(Processor):
    def __init__(self, filepath: str, keys: Optional[str] = None, pad: str = "false"):

        if not H5PY_AVAILABLE:
            raise ImportError("H5Py is not available. Please install H5Py with `pip install h5py`")

        self._file = h5py.File(filepath, "r")
        if keys:
            for k in keys.split("/"):
                self._file = self._file[k]

        self._pad = pad.lower() in ("yes", "true", "t", "1")

    @classmethod
    def typestr(cls):
        return "h5py.dataset"

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
        return torch.from_numpy(self._file[int(value)])


class H5PyMapProcessor(Processor):
    def __init__(self, filepath: str, keys: Optional[str] = None, pad: str = "false"):

        if not H5PY_AVAILABLE:
            raise ImportError("H5Py is not available. Please install H5Py with `pip install h5py`")

        self._file = h5py.File(filepath, "r")
        if keys:
            for k in keys.split("/"):
                self._file = self._file[k]

        self._pad = pad.lower() in ("yes", "true", "t", "1")

    @classmethod
    def typestr(cls):
        return "h5py.map"

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
        return torch.from_numpy(np.array(self._file[value]))
