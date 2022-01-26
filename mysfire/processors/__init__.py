import glob
import importlib
import os
import sys
from typing import Dict, Optional, Type

from ._processor import Processor, S3Processor

# A global registry of all processors available to the system.
PROCESSORS: Dict[str, Type[Processor]] = {}


def register_processor(processor: Type[Processor], typestr: Optional[str] = None) -> Type[Processor]:
    """Register a processor class with the system.

    Args:
        processor (Type[Processor]): The class to register (either as a decorator, or as a class).
        typestr (str, optional): The typestring to register the class with (overrides the class typestring if
                                 specified). Defaults to None.
    """
    PROCESSORS[typestr or processor.typestr()] = processor
    return processor


def register_processor_directory(directory: str) -> None:
    """Register all processors in a directory.

    Args:
        directory (str): The directory to register.
    """
    for _plugin_file in glob.glob(os.path.join(directory, "**", "*.py"), recursive=True):
        _plugin_name = os.path.splitext(os.path.basename(_plugin_file))[0]
        if not _plugin_name.startswith("_") or _plugin_name.startswith("."):  # Ignore hidden files.
            spec = importlib.util.spec_from_file_location(f"mysfire.processors.{_plugin_name}", _plugin_file)
            if spec is not None:
                _plugin_module = importlib.util.module_from_spec(spec)
                sys.modules[_plugin_name] = _plugin_module
                spec.loader.exec_module(_plugin_module)  # type: ignore
            # _plugin_module = importlib.import_module(, "mysfire.processors.{}".format(_plugin_name))
            globals()[_plugin_name] = _plugin_module

    PROCESSORS.update({p.typestr(): p for p in Processor.__subclasses__()})  # type: ignore
    PROCESSORS.update({p.typestr(): p for p in S3Processor.__subclasses__()})  # type: ignore


register_processor_directory(os.path.dirname(__file__))
