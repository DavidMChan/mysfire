from collections import OrderedDict
from typing import List, Optional, Tuple

import torch

from .processors import PROCESSORS, Processor

# Simple import guard to check if pytorch lightning is available before building some of the codebase
PYTORCH_LIGHTNING_AVAILABLE = False
try:
    from pytorch_lightning import LightningDataModule as LDM

    PYTORCH_LIGHTNING_AVAILABLE = True
except ImportError:
    pass


def _build_processor(column: str) -> Tuple[str, Processor]:
    """
    Builds a list of functions that can be used to process a row of data.
    """
    column_name, _, column_type_args = column.partition(":")
    column_type, _, column_args_group = column_type_args.partition(":")

    if column_type not in PROCESSORS:
        raise ValueError(f"Unknown column type: {column_type}")

    if column_args_group:
        # TODO: make this split argument lists better
        column_args = [v.partition("=") for v in column_args_group.split(",")]
        arg_statements = {k: v for k, _, v in column_args}
        return (f"{column_name}+++{column_type}", PROCESSORS[column_type](**arg_statements))
    return (f"{column_name}+++{column_type}", PROCESSORS[column_type]())


def _resolve_samples(filepath: str) -> Tuple[List[List[str]], Optional[List[str]]]:
    with open(filepath, "r") as f:
        samples = [line.strip().split("\t") for line in f]
    return samples[1:], samples[0]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath: str,
        columns: Optional[List[str]] = None,
    ):
        self._filepath = filepath
        self._samples, self._columns = _resolve_samples(self._filepath)
        self._columns = columns if columns is not None else self._columns

        if self._columns is None:
            raise RuntimeError("Dataset {} has no column headers".format(self._filepath))
        self._processors = [_build_processor(c) for c in self._columns]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        row = self._samples[index]
        output = OrderedDict()
        for (k, v), r in zip(self._processors, row):
            output[k] = v(r)
        return output

    def collate_fn(self, batch):
        outputs = {}
        for i, (key, value) in enumerate(batch[0].items()):
            k, _, t = key.rpartition("+++")
            outputs[k] = self._processors[i][1].collate([v[key] for v in batch])
        return outputs


class DataLoader(torch.utils.data.DataLoader):
    # Easy class for managing data loaders
    def __init__(self, filepath: str, columns: Optional[List[str]] = None, **kwargs):
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
