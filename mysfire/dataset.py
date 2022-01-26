from collections import OrderedDict
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple

import torch
from sly.lex import LexError

from .parser import MysfireHeaderLexer, MysfireHeaderParser  # type: ignore
from .processors import PROCESSORS, Processor

# Simple import guard to check if pytorch lightning is available before building some of the codebase
PYTORCH_LIGHTNING_AVAILABLE = False
try:
    from pytorch_lightning import LightningDataModule as LDM

    PYTORCH_LIGHTNING_AVAILABLE = True
except ImportError:
    pass


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

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return OrderedDict((k, v(r)) for (k, v), r in zip(self._processors, self._samples[index]))

    def collate_fn(self, batch: List[Mapping[str, Any]]) -> Dict[str, Any]:
        return {key: self._processors[i][1].collate([v[key] for v in batch]) for i, key in enumerate(batch[0].keys())}

    def get_processors(self) -> List[Tuple[str, Processor]]:
        return self._processors


class DataLoader(torch.utils.data.DataLoader):
    # Easy class for managing data loaders
    def __init__(self, filepath: str, columns: Optional[List[str]] = None, **kwargs) -> None:
        self._active_dataset = Dataset(filepath, columns)
        if "collate_fn" not in kwargs:
            kwargs.update(
                {"collate_fn": self._active_dataset.collate_fn}
            )  # Update the collate function with the correct function if it doesn't exist
        super().__init__(self._active_dataset, **kwargs)


if PYTORCH_LIGHTNING_AVAILABLE:

    class LightningDataModule(LDM):
        def __init__(
            self,
            train_filepath: Optional[str] = None,
            train_columns: Optional[List[str]] = None,
            val_filepath: Optional[str] = None,
            val_columns: Optional[List[str]] = None,
            test_filepath: Optional[str] = None,
            test_columns: Optional[List[str]] = None,
            train_batch_size: int = 32,
            val_batch_size: int = 32,
            test_batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
        ):
            super().__init__()

            self._train_filepath = train_filepath
            self._train_columns = train_columns
            self._val_filepath = val_filepath
            self._val_columns = val_columns
            self._test_filepath = test_filepath
            self._test_columns = test_columns

            self._train_batch_size = train_batch_size
            self._val_batch_size = val_batch_size
            self._test_batch_size = test_batch_size
            self._num_workers = num_workers
            self._pin_memory = pin_memory

        def setup(self, stage=None):
            if stage in ("fit", None):
                self._train_ds = Dataset(self._train_filepath, self._train_columns) if self._train_filepath else None
                self._val_ds = Dataset(self._val_filepath, self._val_columns) if self._val_filepath else None
            if stage in ("test", None):
                self._val_ds = Dataset(self._val_filepath, self._val_columns) if self._val_filepath else None
                self._test_ds = Dataset(self._test_filepath, self._test_columns) if self._test_filepath else None

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                self._train_ds,
                shuffle=True,
                collate_fn=self._train_ds.collate_fn,
                batch_size=self._train_batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )

        def val_dataloader(self):
            return torch.utils.data.DataLoader(
                self._val_ds,
                shuffle=True,
                collate_fn=self._val_ds.collate_fn,
                batch_size=self._val_batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )

        def test_dataloader(self):
            return torch.utils.data.DataLoader(
                self._test_ds,
                shuffle=True,
                collate_fn=self._test_ds.collate_fn,
                batch_size=self._test_batch_size,
                num_workers=self._num_workers,
                pin_memory=self._pin_memory,
            )
