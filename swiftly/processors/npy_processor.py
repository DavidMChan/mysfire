from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..s3_utils import resolve_s3_or_local
from ._array_utils import stack_arrays_as_dict
from ._processor import S3Processor


class NpyProcessor(S3Processor):
    def __init__(
        self,
        pad: str = "false",
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_region: Optional[str] = None,
    ) -> None:

        super().__init__(
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_region=s3_region,
        )

        self._pad = pad.lower() in ("yes", "true", "t", "1")

    @classmethod
    def typestr(cls):
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
        with resolve_s3_or_local(value, connection=self.s3_client) as f:
            return torch.from_numpy(np.load(f))


class NpyIndexedFileProcessor(S3Processor):
    def __init__(
        self,
        filepath: str,
        pad: str = "false",
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_region: Optional[str] = None,
    ) -> None:

        super().__init__(
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_region=s3_region,
        )

        with resolve_s3_or_local(filepath, connection=self.s3_client) as f:
            self._data = np.load(f)
        self._pad = pad.lower() in ("yes", "true", "t", "1")

    @classmethod
    def typestr(cls):
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
        return torch.from_numpy(self._data(int(value))) if value else None
