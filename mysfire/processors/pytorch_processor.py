from typing import Dict, List, Optional, Union

import torch

from ..s3_utils import resolve_s3_or_local
from ._array_utils import stack_arrays_as_dict
from ._processor import S3Processor


class PtProcessor(S3Processor):
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
        return "pt"

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
            return torch.load(f, map_location=torch.device("cpu"))
