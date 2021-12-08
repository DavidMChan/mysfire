from typing import Dict, List, Optional, Union

import torch

from ..torch_utils import padded_stack


def stack_arrays_as_dict(
    batch: List[Optional[torch.Tensor]], pad: bool = True
) -> Optional[
    Union[
        torch.Tensor,
        Dict[str, Union[Optional[torch.Tensor], Optional[List[Optional[torch.Tensor]]]]],
        List[Optional[torch.Tensor]],
    ]
]:
    # If the input shape is the same for every element in the batch, stack the arrays, else pad the arrays to the
    # same shape.
    proto_s = [b for b in batch if b is not None]
    if not proto_s:
        return None
    proto = proto_s[0]

    if all(x is None or x.shape == proto.shape for x in batch):
        if pad:
            return {
                "data": torch.stack(
                    [x if x is not None else torch.zeros_like(proto) for x in batch],
                    dim=0,
                ),
                "seqlen": torch.tensor([x.shape[0] if x is not None else 0 for x in batch]),
            }

        else:
            return torch.stack(
                [x if x is not None else torch.zeros_like(proto) for x in batch],
                dim=0,
            )

    if all(x is None or x.shape[1:] == proto.shape[1:] for x in batch) and pad:
        # Pad the first axis, and return sequence lengths
        tensors = [x if x is not None else torch.zeros(*proto.shape[1:]).to(proto.dtype) for x in batch]
        d, s = padded_stack(tensors)
        return {"data": d, "seqlen": s}

    # TODO: Correct the return types on this data
    if pad:
        return {"data": batch, "seqlen": torch.tensor([x.shape[0] if x is not None else 0 for x in batch])}
    return batch
