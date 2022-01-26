from typing import List, Optional

import torch

from .dataset import Dataset

# Simple import guard to check if pytorch lightning is available before building some of the codebase
PYTORCH_LIGHTNING_AVAILABLE = False
try:
    from pytorch_lightning import LightningDataModule as LDM

    PYTORCH_LIGHTNING_AVAILABLE = True
except ImportError:
    pass

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
