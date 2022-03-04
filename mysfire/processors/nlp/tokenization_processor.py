from collections import Counter
from typing import Any, Dict, List, Optional, Iterable
from typing import Counter as CounterType


import json
import torch

from mysfire.torch_utils import padded_stack
from mysfire.processors._processor import Processor, S3Processor

HUGGINGFACE_TOKENIZERS_AVAILABLE = False
try:
    from tokenizers import Tokenizer

    HUGGINGFACE_TOKENIZERS_AVAILABLE = True
except ImportError:
    pass

HUGGINGFACE_TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer

    HUGGINGFACE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class HuggingfaceTokenizationProcessor(Processor):
    def __init__(
        self,
        tokenizer_json: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        if not HUGGINGFACE_TOKENIZERS_AVAILABLE:
            raise ImportError(
                "Huggingface tokenizers are not available."
                " Please install Huggingface tokenizers with `pip install tokenizers`"
            )

        super().__init__(**kwargs)

        # Load the tokenizer definitions from JSON file
        if tokenizer_json is None:
            raise ValueError("tokenizer_json must be provided to Huggingface Tokenization processor")
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


class TransformersTokenizationProcessor(Processor):
    def __init__(
        self,
        tokenizer: Optional[str] = None,
        **kwargs: Any,
    ) -> None:

        if not HUGGINGFACE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Huggingface transformers are not available."
                " Please install Huggingface transformers with `pip install transformers`"
            )

        super().__init__(**kwargs)

        # Load the tokenizer definitions from JSON file
        if tokenizer is None:
            raise ValueError("tokenizer must be provided to Huggingface Tokenization processor")
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    @classmethod
    def typestr(cls) -> str:
        return "nlp.transformers_tokenizer"

    def collate(self, batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        tokens, tokens_seqlen = padded_stack([x["tokens"] if x else [] for x in batch])
        return {
            "text": [x["text"] if x else "" for x in batch],
            "__root__": tokens,
            "seqlen": tokens_seqlen,
            "tokens_text": [x["tokens_text"] if x else [] for x in batch],
        }

    def __call__(self, value: str) -> Optional[Dict[str, Any]]:
        tokens = self._tokenizer(value.strip().lower())
        return {
            "text": value,
            "tokens": torch.LongTensor(tokens.input_ids),
            "tokens_text": self._tokenizer.convert_ids_to_tokens(tokens.input_ids),
        }


class VocabTokenizationProcessor(S3Processor):
    def __init__(
        self,
        vocab_json: str,
        max_sequence_length: int,
        unk_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Simple whitepsace-based tokenization using a JSON which maps strings to integers.

        Args:
            vocab_json (str): Path to a JSON file which maps strings to integers.
            max_sequence_length (int): Maximum sequence length.
        """

        super().__init__(**kwargs)

        # Load the tokenizer definitions from JSON file
        with self.resolve_to_local(vocab_json) as jf:
            with open(jf, "r") as f:
                self._vocab = json.load(f)

        # Handle some special token definitions
        if unk_token is None:
            unk_token = "<unk>"
        if unk_token not in self._vocab:
            raise ValueError(f"unk_token {unk_token} not in vocab")
        if pad_token is None:
            pad_token = "<pad>"
        if pad_token not in self._vocab:
            raise ValueError(f"pad_token {pad_token} not in vocab")

        self._unk_token = unk_token
        self._pad_token = pad_token

        self._max_sequence_length = max_sequence_length
        self._vocab_size = max(self._vocab.values()) + 1

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @classmethod
    def typestr(cls) -> str:
        return "nlp.vocab_tokenization"

    @classmethod
    def build_vocab_from_strings(cls, outut_path: str, input_strings: Iterable[str], vocab_size: int = 10000) -> None:
        """Build a vocab to a JSON file from a list of input strings.

        Args:
            outut_path (str): THe output path to write the JSON file to
            input_strings (Iterable[str]): The strings to build the vocabulary from
        """

        base_tokens = {
            "<pad>": 0,
            "<unk>": 1,
        }

        counts: CounterType[str] = Counter()
        n_strings = 0
        for s in input_strings:
            counts.update(s.split())
            n_strings += 1

        most_common = counts.most_common(vocab_size - len(base_tokens))
        base_tokens.update({k: i + len(base_tokens) for i, (k, v) in enumerate(most_common)})

        with open(outut_path, "w") as f:
            json.dump(base_tokens, f)

        print("Constructed vocab of size {} from {} strings".format(len(base_tokens), n_strings))

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "tokens": torch.stack([x["tokens"] for x in batch], dim=0),
            "sequence_mask": torch.stack([x["sequence_mask"] for x in batch], dim=0),
            "text": [x["text"] for x in batch],
        }

    def __call__(self, value: str) -> Dict[str, Any]:

        tokens = value.split()
        if len(tokens) >= self._max_sequence_length:
            tokens = tokens[: self._max_sequence_length]
            sequence_mask = torch.ones(self._max_sequence_length)
        else:
            # Pad the sequence with pad tokens
            tokens = tokens + [self._pad_token] * (self._max_sequence_length - len(tokens))
            sequence_mask = torch.cat([torch.ones(len(tokens)), torch.zeros(self._max_sequence_length - len(tokens))])

        # Convert tokens to integers
        numerized = torch.LongTensor([self._vocab.get(t, self._vocab[self._unk_token]) for t in tokens])

        return {
            "text": tokens,
            "tokens": numerized,
            "sequence_mask": sequence_mask,
        }
