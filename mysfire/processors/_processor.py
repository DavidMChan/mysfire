from typing import Any, List, Optional, Protocol

from ..s3_utils import Connection


class Processor(Protocol):
    """Core processor type - Processores must have three methods:

    - __call__: Takes a single path/string from the TSV and returns a single object
    - collate: Takes a list of objects and returns a single object
    - typestr: Return a string which is matched against the type field of the TSV header

    Optionally, processors may have a `__init__` method which takes any set of Optional[str] arguments
    which are parsed from the TSV header.

    Args:
        Protocol ([type]): [description]
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def typestr(cls):
        raise NotImplementedError()

    def collate(self, batch: List[Any]) -> Any:
        raise NotImplementedError()

    def __call__(self, column: str) -> Optional[Any]:
        raise NotImplementedError()


class S3Processor(Processor):
    """
    Processor base class which additionally adds a convenient connection to allow supporting (optional) S3 objects
    """

    @classmethod
    def typestr(cls):
        return "__s3"

    def __init__(
        self,
        s3_endpoint: Optional[str] = None,
        s3_access_key: Optional[str] = None,
        s3_secret_key: Optional[str] = None,
        s3_region: Optional[str] = None,
    ) -> None:

        # Extra variables for enabling S3 support
        self._s3_endpoint = s3_endpoint
        self._s3_access_key = s3_access_key
        self._s3_secret_key = s3_secret_key
        self._s3_region = s3_region
        self._s3_client = None
        if self._s3_endpoint or self._s3_access_key or self._s3_secret_key or self._s3_region:
            assert self._s3_access_key is not None, "S3 access key is required"
            assert self._s3_secret_key is not None, "S3 secret key is required"

            self._s3_client = Connection(
                access_key=self._s3_access_key,
                secret_key=self._s3_secret_key,
                endpoint=self._s3_endpoint,
                region=self._s3_region,
            )

    @property
    def s3_client(self) -> Optional[Connection]:
        return self._s3_client
