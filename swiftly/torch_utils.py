from typing import Sequence, Tuple

import torch


def padded_stack(
    tensor_list: Sequence[torch.Tensor], padding_value: int = 0, after: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Stacks a set of tensors that differ only in the first dimension. Returns the padded tensors, and the sequence
    lengths.
    """
    padded_tensors = []
    sequence_lengths = []
    max_len = max(t.shape[0] for t in tensor_list)
    base_padding = (padding_value * torch.ones_like(tensor_list[0][0])).unsqueeze(0)
    for elem in tensor_list:
        padding_len = max_len - elem.shape[0]
        sequence_lengths.append(elem.shape[0])
        padding = base_padding.expand(padding_len, *[-1 for _ in range(len(elem.shape) - 1)])
        if after:
            padded_tensors.append(torch.cat([elem, padding], dim=0))
        else:
            padded_tensors.append(torch.cat([padding, elem], dim=0))

    stacked_tensors = torch.stack(padded_tensors, dim=0)
    return stacked_tensors, torch.LongTensor(sequence_lengths).to(stacked_tensors.device)
