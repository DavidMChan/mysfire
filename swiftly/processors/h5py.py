from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import torch

from ._array_utils import stack_arrays_as_dict
from ._processor import Processor


class H5PyDatasetProcessor(Processor):
    def __init__(self, filepath: str, keys: Optional[str] = None, pad: str = "false"):
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
    ) -> Union[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        return stack_arrays_as_dict(batch, self._pad)

    def __call__(self, value: str) -> Optional[torch.Tensor]:
        return torch.from_numpy(self._file[int(value)])


class H5PyMapProcessor(Processor):
    def __init__(self, filepath: str, keys: Optional[str] = None, pad: str = "false"):
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
    ) -> Union[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        return stack_arrays_as_dict(batch, self._pad)

    def __call__(self, value: str) -> Optional[torch.Tensor]:
        return torch.from_numpy(np.array(self._file[value]))
