from typing import List, Optional

import torch

from ._processor import Processor


class IntProcessor(Processor):
    @classmethod
    def typestr(cls):
        return "int"

    def collate(self, batch: List[Optional[int]]) -> torch.Tensor:
        return torch.tensor(batch)

    def __call__(self, value: str) -> Optional[int]:
        return int(value) if value else None


class FloatProcessor(Processor):
    @classmethod
    def typestr(cls):
        return "float"

    def collate(self, batch: List[Optional[float]]) -> torch.Tensor:
        return torch.tensor(batch)

    def __call__(self, value: str) -> Optional[float]:
        return float(value) if value else None


class StringProcessor(Processor):
    @classmethod
    def typestr(cls):
        return "str"

    def collate(self, batch: List[Optional[str]]) -> List[str]:
        return [b or "" for b in batch]

    def __call__(self, value: str) -> Optional[str]:
        return value or None


class StringListProcessor(Processor):
    def __init__(self, delimiter: str = "###"):
        self._delimiter = delimiter

    @classmethod
    def typestr(cls):
        return "str.list"

    def collate(self, batch: List[Optional[List[str]]]) -> List[List[str]]:
        return [b or [] for b in batch]

    def __call__(self, value: str) -> Optional[List[str]]:
        return value.split(self._delimiter) if value else None
