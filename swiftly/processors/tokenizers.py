from typing import Any, Dict, List, Optional

import torch

from ..torch_utils import padded_stack
from ._processor import Processor

HUGGINGFACE_TOKENIZERS_AVAILABLE = False
try:
    from tokenizers import Tokenizer

    HUGGINGFACE_TOKENIZERS_AVAILABLE = True
except ImportError:
    pass


class HuggingfaceTokenizationProcessor(Processor):
    def __init__(
        self,
        tokenizer_json: Optional[str] = None,
    ) -> None:

        if not HUGGINGFACE_TOKENIZERS_AVAILABLE:
            raise ImportError(
                "Huggingface tokenizers are not available."
                " Please install Huggingface tokenizers with `pip install tokenizers`"
            )

        # Load the tokenizer definitions from JSON file
        print(tokenizer_json)
        self._tokenizer = Tokenizer.from_file(tokenizer_json)

    @classmethod
    def typestr(cls) -> str:
        return "nlp.huggingface_tokenization"

    def collate(self, batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        numerized, numerized_seqlen = padded_stack([x["numerized"] if x else [] for x in batch])
        return {
            "text": [x["text"] if x else "" for x in batch],
            "tokens": [x["tokens"] if x else [] for x in batch],
            "numerized": numerized,
            "numerized_seqlen": numerized_seqlen,
        }

    def __call__(self, value: str) -> Optional[Dict[str, Any]]:
        tokens = self._tokenizer.encode(value.strip().lower())
        return {
            "text": value,
            "tokens": tokens.tokens,
            "numerized": torch.IntTensor(tokens.ids),
        }
