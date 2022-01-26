__version__ = "0.0.1"


from .dataset import DataLoader, Dataset  # noqa: F401
from .one_shot import OneShotLoader  # noqa: F401
from .processors import Processor, S3Processor, register_processor, register_processor_directory  # noqa: F401

# Import the lightning module if available
try:
    from .lightning import LightningDataModule  # noqa: F401
except ImportError:
    pass
