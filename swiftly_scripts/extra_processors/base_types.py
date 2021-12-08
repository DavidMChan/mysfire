from typing import List, Optional

import torch

from ._processor import Processor


class ExtraIntProcessor(Processor):
    @classmethod
    def typestr(cls):
        return "int"

    def collate(self, batch: List[Optional[str]]) -> List[str]:
        return [b or "" for b in batch]

    def __call__(self, value: str) -> Optional[str]:
        return "OMG!!! " + value if value else None
