import torch

from mysfire import DataLoader

dataloader = DataLoader(
    "/home/davidchan/Projects/mysfire/mysfire_scripts/test_s3.tsv",
    batch_size=2,
    shuffle=False,
)
for batch in dataloader:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                print(kk, vv.shape)
        else:
            print(k, v)
