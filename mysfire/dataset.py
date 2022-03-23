import logging
import random
from typing import Any, Dict, Generator, List, Mapping, Optional, Set, Tuple

import pyarrow as pa
import torch
from sly.lex import LexError

from .cloud_utils import resolve_to_local_path
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
    # TODO: Allow passing connection details through to this function
    with resolve_to_local_path(filepath) as f:
        with open(f, "r") as f:
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
        self, filepath: str, columns: Optional[List[str]] = None, resample_on_processor_exception: bool = False
    ) -> None:
        self._filepath = filepath
        samples, columns = resolve_samples(self._filepath)
        logging.info(f"Loaded {len(samples)} samples from {self._filepath}")

        self._columns = pa.array(columns) if columns else None
        self._samples = pa.array(samples)
        self._resample_on_exception = resample_on_processor_exception

        if self._columns is None:
            raise RuntimeError("Dataset {} has no column headers".format(self._filepath))
        self._processors = list(build_processors("\t".join(columns or [])))
        logging.info(f"Loaded {len(self._processors)} processors")

        # Pre-validate the samples
        for i, sample in enumerate(samples):
            if len(sample) != len(self._processors):
                raise RuntimeError(
                    "Line {} in dataset {} has {} columns, but {} columns were expected".format(
                        i + 1, filepath, len(sample), len(self._processors)
                    )
                )

        # Initialize the variables
        self.vars = VariableRegistry()
        for key, processor in self._processors:
            self.vars.add_processor(key, processor)

        # Run pre-initilization on the processors (for computing global values)
        logging.info("Running pre-initialization on processors")
        for i, (_, processor) in enumerate(self._processors):
            if hasattr(processor, "validate_samples"):
                processor.validate_samples([s[i] for s in samples])  # type: ignore
            if hasattr(processor, "pre_init"):
                processor.pre_init([s[i] for s in samples])  # type: ignore

        logging.info("Dataset loaded successfully")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        tested_indices: Set[int] = set()
        while len(tested_indices) < len(self._samples):
            try:
                return {k: v.load(r.as_py()) for (k, v), r in zip(self._processors, self._samples[index])}
            except Exception as e:
                if not self._resample_on_exception:
                    raise e from e
                tested_indices.add(index)
                index = random.choice(tuple(set(range(len(self._samples))) - tested_indices))
                logging.warning(f"Resampling row {index} due to data-loading exception: {e}")

        raise RuntimeError("Could not load data for any row!")

    def collate_fn(self, batch: List[Mapping[str, Any]]) -> Dict[str, Any]:
        return _flatten_dicts(
            {key: self._processors[i][1].collate([v[key] for v in batch]) for i, key in enumerate(batch[0].keys())}
        )

    def get_processors(self) -> List[Tuple[str, Processor]]:
        return self._processors


class DataLoader(torch.utils.data.DataLoader):
    # Easy class for managing data loaders
    def __init__(
        self,
        filepath: str,
        columns: Optional[List[str]] = None,
        resample_on_processor_exception: bool = True,
        **kwargs: Any,
    ) -> None:
        _active_dataset = Dataset(filepath, columns, resample_on_processor_exception)
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = _active_dataset.collate_fn

        super().__init__(_active_dataset, **kwargs)
