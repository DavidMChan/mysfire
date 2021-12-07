from typing import Any, List, Optional, Protocol


class Processor(Protocol):
    def __init__(*args, **kwargs) -> None:
        pass

    @classmethod
    def typestr(cls):
        raise NotImplementedError()

    def collate(self, batch: List[Optional[Any]]) -> Any:
        raise NotImplementedError()

    def __call__(self, column: str) -> Optional[Any]:
        raise NotImplementedError()
