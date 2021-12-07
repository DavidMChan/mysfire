import logging
import tempfile
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from ..s3_utils import Connection
from ._array_utils import stack_arrays_as_dict
from ._processor import Processor


class NpyProcessor(Processor):
    def __init__(
        self,
        pad: str = "false",
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_region: Optional[str] = None,
    ) -> None:
        self._pad = pad.lower() in ("yes", "true", "t", "1")

        # Extra variables for enabling S3 support
        self._s3_endpoint = s3_endpoint
        self._s3_access_key = s3_access_key
        self._s3_secret_key = s3_secret_key
        self._s3_region = s3_region
        self._s3_client = None
        if self._s3_endpoint or self._s3_access_key or self._s3_secret_key or self._s3_region:
            assert self._s3_access_key is not None, "S3 access key is required"
            assert self._s3_secret_key is not None, "S3 secret key is required"

            self._s3_client = Connection(
                access_key=self._s3_access_key,
                secret_key=self._s3_secret_key,
                endpoint=self._s3_endpoint,
                region=self._s3_region,
            )

    @classmethod
    def typestr(cls):
        return "npy"

    def collate(
        self, batch: List[Optional[torch.Tensor]]
    ) -> Union[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        return stack_arrays_as_dict(batch, self._pad)

    def _load_from_s3(self, path: str) -> np.ndarray:
        bucket, _, key = path[5:].partition("/")
        if self._s3_client is None:
            raise ValueError("S3 client not initialized")

        with tempfile.NamedTemporaryFile() as f:
            for i in range(5):  # Retry with backoff
                try:
                    self._s3_client.download(key, f.name, bucket)
                    return np.load(f.name)
                except Exception as e:
                    logging.warning(f"Failed to load {path}: {e}")
                    time.sleep(2 ** i)

        raise ValueError(f"Failed to load {path} from S3")

    def __call__(self, value: str) -> Optional[torch.Tensor]:
        try:
            if value.lower().startswith("s3://"):
                return torch.from_numpy(self._load_from_s3(value))
            return torch.from_numpy(np.load(value)) if value else None
        except Exception as e:
            logging.error(f"Failed to load npy file {value}: {e}")
            return None


class NpyIndexedFileProcessor(Processor):
    def __init__(self, filepath: str, pad: str = "false") -> None:
        self._data = np.load(filepath)
        self._pad = pad.lower() in ("yes", "true", "t", "1")

    @classmethod
    def typestr(cls):
        return "npy.indexed_file"

    def collate(
        self, batch: List[Optional[torch.Tensor]]
    ) -> Union[Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        return stack_arrays_as_dict(batch, self._pad)

    def __call__(self, value: str) -> Optional[torch.Tensor]:
        return torch.from_numpy(self._data(int(value))) if value else None
