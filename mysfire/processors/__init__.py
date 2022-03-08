# import glob
# import importlib
# import os
# import sys
from typing import Dict, Optional, Type

# import itertools

from ._processor import Processor, S3Processor  # noqa: F401

# A global registry of all processors available to the system.
PROCESSORS: Dict[str, Type[Processor]] = {}


def register_processor(processor: Type[Processor], typestr: Optional[str] = None) -> Type[Processor]:
    """Register a processor class with the system.

    Args:
        processor (Type[Processor]): The class to register (either as a decorator, or as a class).
        typestr (str, optional): The typestring to register the class with (overrides the class typestring if
                                 specified). Defaults to None.
    """
    PROCESSORS[typestr or processor.typestr()] = processor  # type: ignore
    return processor


# def register_processor_directory(directory: str) -> None:
#     """Register all processors in a directory.

#     Args:
#         directory (str): The directory to register.
#     """
#     for _plugin_file in glob.glob(os.path.join(directory, "**", "*.py"), recursive=True):
#         _plugin_name = os.path.splitext(os.path.basename(_plugin_file))[0]
#         if not _plugin_name.startswith("_") or _plugin_name.startswith("."):  # Ignore hidden files.
#             spec = importlib.util.spec_from_file_location(f"mysfire.processors.{_plugin_name}", _plugin_file)
#             if spec is not None:
#                 _plugin_module = importlib.util.module_from_spec(spec)
#                 sys.modules[_plugin_name] = _plugin_module
#                 spec.loader.exec_module(_plugin_module)  # type: ignore
#                 globals()[_plugin_name] = _plugin_module

#     for p in itertools.chain(Processor.__subclasses__(), S3Processor.__subclasses__()):
#         try:
#             PROCESSORS[p.typestr()] = p  # type: ignore
#         except NotImplementedError as e:
#             if p.__name__ in ("S3Processor", "Processor"):
#                 continue
#             raise e from e

# PROCESSORS.update({p.typestr(): p for p in Processor.__subclasses__()})  # type: ignore
# PROCESSORS.update({p.typestr(): p for p in S3Processor.__subclasses__()})


# register_processor_directory(os.path.dirname(__file__))
from .base_types_processor import IntProcessor, FloatProcessor, StringProcessor, StringListProcessor  # noqa: F401, E402
from .h5py_processor import H5PyDatasetProcessor, H5PyMapProcessor  # noqa: F401, E402
from .image_processor import ImageProcessor  # noqa: F401, E402
from .npy_processor import NpyProcessor, NpyIndexedFileProcessor  # noqa: F401, E402
from .pytorch_processor import PtProcessor  # noqa: F401, E402
from .video_processor import VideoProcessor, FixedSizeOutputVideoProcessor  # noqa: F401, E402
from .nlp.tokenization_processor import (  # noqa: F401, E402
    HuggingfaceTokenizationProcessor,
    TransformersTokenizationProcessor,
    VocabTokenizationProcessor,
)
