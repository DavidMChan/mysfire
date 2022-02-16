from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple

import torch
import pyarrow as pa
from sly.lex import LexError

from .parser import MysfireHeaderLexer, MysfireHeaderParser  # type: ignore
from .processors import PROCESSORS, Processor
from .vars_registry import VariableRegistry


def build_processors(header_line: str) -> Generator[Tuple[str, Processor], None, None]:
    lexer = MysfireHeaderLexer()
    parser = MysfireHeaderParser()
    error = False
    processor_defs = None

    try:
        processor_defs = parser.parse(lexer.tokenize(header_line))
    except LexError as e:
        error = e
    if error:
        # Avoids an annoying error message with multiple layers
        raise SyntaxError(f"Invalid header line: {error}")

    for output_key, proc_type, args in processor_defs or []:
        if proc_type not in PROCESSORS:
            raise ValueError(f"Unknown column type: {proc_type}")
        processor = PROCESSORS[proc_type](**(args or {}))
        yield (output_key, processor)


def resolve_samples(filepath: str) -> Tuple[List[List[str]], Optional[List[str]]]:
    with open(filepath, "r") as f:
        samples = [line.strip().split("\t") for line in f]
    return samples[1:], samples[0]


def _flatten_dicts(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in list(dictionary.items()):
        if isinstance(value, dict):
            dictionary.pop(key)
            for kk, vv in value.items():
                dictionary[key if kk == "__root__" else f"{key}.{kk}"] = vv
    return dictionary


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath: str,
        columns: Optional[List[str]] = None,
    ) -> None:
        self._filepath = filepath
        samples, columns = resolve_samples(self._filepath)
        self._columns = pa.array(columns) if columns else None
        self._samples = pa.array(samples)

        if self._columns is None:
            raise RuntimeError("Dataset {} has no column headers".format(self._filepath))
        self._processors = list(build_processors("\t".join(columns or [])))

        # Initialize the variables
        self.vars = VariableRegistry()
        for key, processor in self._processors:
            self.vars.add_processor(key, processor)

        # Run pre-initilization on the processors (for computing global values)
        for i, (_, processor) in enumerate(self._processors):
            if hasattr(processor, "pre_init"):
                processor.pre_init([s[i] for s in samples])  # type: ignore

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return {k: v(r.as_py()) for (k, v), r in zip(self._processors, self._samples[index])}

    def collate_fn(self, batch: List[Mapping[str, Any]]) -> Dict[str, Any]:
        return _flatten_dicts(
            {key: self._processors[i][1].collate([v[key] for v in batch]) for i, key in enumerate(batch[0].keys())}
        )

    def get_processors(self) -> List[Tuple[str, Processor]]:
        return self._processors


class DataLoader(torch.utils.data.DataLoader):
    # Easy class for managing data loaders
    def __init__(self, filepath: str, columns: Optional[List[str]] = None, **kwargs: Any) -> None:
        _active_dataset = Dataset(filepath, columns)
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = _active_dataset.collate_fn

        super().__init__(_active_dataset, **kwargs)
