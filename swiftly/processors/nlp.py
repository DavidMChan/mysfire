from typing import Any, Dict, List, Optional

import torch
from tokenizers import Tokenizer

from ..torch_utils import padded_stack
from ._processor import Processor


class HuggingfaceTokenizationProcessor(Processor):
    def __init__(
        self,
        tokenizer_json: Optional[str] = None,
    ) -> None:
        # Load the tokenizer definitions from JSON file
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
