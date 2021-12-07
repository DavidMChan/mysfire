import importlib
import logging
import os
from typing import Dict, Type

from ._processor import Processor

# Dynamically import all processors in this directory
_plugin_dir = os.path.dirname(__file__)
_plugin_files = os.listdir(_plugin_dir)
for _plugin_file in _plugin_files:
    if _plugin_file.endswith(".py") and not _plugin_file.startswith("_"):
        _plugin_name = _plugin_file[:-3]
        _plugin_module = importlib.import_module("swiftly.processors.{}".format(_plugin_name))
        globals()[_plugin_name] = _plugin_module

PROCESSORS: Dict[str, Type[Processor]] = {p.typestr(): p for p in Processor.__subclasses__()}  # type: ignore
for k, v in PROCESSORS.items():
    logging.debug("Loaded Processor: {} for type {}".format(k, v))
