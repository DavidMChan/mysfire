import sys

import torch

from swiftly import Dataset

x = Dataset("/home/davidchan/Projects/swiftly/swiftly_scripts/simple_data.tsv")
dataloader = torch.utils.data.DataLoader(x, batch_size=2, shuffle=False, collate_fn=x.collate_fn)
for batch in dataloader:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, v)
