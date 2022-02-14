from .processors._processor import Processor
from typing import Any, Sequence, Dict


class VarProxy:
    def __init__(self, processor: Processor, attrs: Sequence[str]) -> None:
        self._proxy_fns = {}
        self._processor = processor
        for attr in attrs:
            self._proxy_fns[attr] = lambda: getattr(processor, attr)

    def __getattr__(self, __name: str) -> Any:
        if __name in self._proxy_fns:
            return self._proxy_fns[__name]()  # type: ignore
        raise ValueError(
            f"""{__name} is not a variable exported by {self._processor.__class__.__name__}."""
            f""" Valid variables are: {', '.join(self._proxy_fns.keys())}"""
        )


class VariableRegistry:
    def __init__(self) -> None:
        self._proxies: Dict[str, VarProxy] = {}

    def add_processor(self, key: str, processor: Processor) -> None:
        # Discover all properties of the processor, and link them to the registry
        processor_properties = [name for name, value in vars(type(processor)).items() if isinstance(value, property)]
        self._proxies[key] = VarProxy(processor, processor_properties)

    def __getattr__(self, __name: str) -> Any:
        if __name in self._proxies:
            return self._proxies[__name]

        raise ValueError(
            f"{__name} does not refer to a column in this dataset. Valid columns are: {', '.join(self._proxies.keys())}"
        )
