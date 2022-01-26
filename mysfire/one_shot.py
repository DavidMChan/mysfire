from collections import OrderedDict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .dataset import Dataset, build_processors, resolve_samples
from .processors import Processor


class OneShotLoader:
    def __init__(
        self,
        definition: Optional[str] = None,
        filepath: Optional[str] = None,
        columns: Optional[Iterable[str]] = None,
        processors: Optional[Iterable[Tuple[str, Processor]]] = None,
    ) -> None:

        if definition is not None:
            self._processors = list(build_processors(definition))
        elif columns is not None:
            self._processors = list(build_processors("\t".join(columns)))
        elif processors is not None:
            self._processors = list(processors)
        elif filepath is not None:
            _, columns = resolve_samples(filepath)
            if columns is None:
                raise RuntimeError("OneShotLoader requires a file with column headers")
            self._processors = list(build_processors("\t".join(columns)))
        else:
            raise RuntimeError("OneShotLoader requires either a definition, columns, or filepath")

    def __call__(self, samples: Sequence[Sequence[str]]) -> Mapping[str, Any]:
        if len(samples) == 0:
            raise RuntimeError("OneShotLoader requires at least one sample")
        if len(samples) == 1:
            return OrderedDict((k, v(r)) for (k, v), r in zip(self._processors, samples[0]))

        return self.collate_fn(
            [OrderedDict((k, v(r)) for (k, v), r in zip(self._processors, sample)) for sample in samples]
        )

    def collate_fn(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return {key: self._processors[i][1].collate([v[key] for v in batch]) for i, key in enumerate(batch[0].keys())}

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "OneShotLoader":
        return cls(processors=dataset.get_processors())
