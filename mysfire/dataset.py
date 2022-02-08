from collections import OrderedDict
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple

import torch
from sly.lex import LexError

from .parser import MysfireHeaderLexer, MysfireHeaderParser  # type: ignore
from .processors import PROCESSORS, Processor


def build_processors(header_line: str) -> Generator[Tuple[str, Processor], None, None]:
    lexer = MysfireHeaderLexer()
    parser = MysfireHeaderParser()
    error = False
    try:
        processor_defs = parser.parse(lexer.tokenize(header_line))
    except LexError as e:
        error = e
    if error:
        # Avoids an annoying error message with multiple layers
        raise SyntaxError(f"Invalid header line: {error}")

    for output_key, proc_type, args in processor_defs:
        if proc_type not in PROCESSORS:
            raise ValueError(f"Unknown column type: {proc_type}")
        yield (output_key, PROCESSORS[proc_type](**(args or {})))


def resolve_samples(filepath: str) -> Tuple[List[List[str]], Optional[List[str]]]:
    with open(filepath, "r") as f:
        samples = [line.strip().split("\t") for line in f]
    return samples[1:], samples[0]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath: str,
        columns: Optional[List[str]] = None,
    ) -> None:
        self._filepath = filepath
        self._samples, self._columns = resolve_samples(self._filepath)
        self._columns = columns if columns is not None else self._columns

        if self._columns is None:
            raise RuntimeError("Dataset {} has no column headers".format(self._filepath))
        self._processors = list(build_processors("\t".join(self._columns)))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return OrderedDict((k, v(r)) for (k, v), r in zip(self._processors, self._samples[index]))

    def collate_fn(self, batch: List[Mapping[str, Any]]) -> Dict[str, Any]:
        return {key: self._processors[i][1].collate([v[key] for v in batch]) for i, key in enumerate(batch[0].keys())}

    def get_processors(self) -> List[Tuple[str, Processor]]:
        return self._processors


class DataLoader(torch.utils.data.DataLoader):
    # Easy class for managing data loaders
    def __init__(self, filepath: str, columns: Optional[List[str]] = None, **kwargs: Any) -> None:
        self._active_dataset = Dataset(filepath, columns)
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = self._active_dataset.collate_fn
        super().__init__(self._active_dataset, **kwargs)
