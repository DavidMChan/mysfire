import torch

from swiftly import DataLoader

dataloader = DataLoader(
    "/home/davidchan/Projects/swiftly/swiftly_scripts/test_video.tsv",
    batch_size=2,
    shuffle=False,
)
for batch in dataloader:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    for kkk, vvv in vv.items():
                        print(kkk, vvv.shape if isinstance(vvv, torch.Tensor) else vvv)
                else:
                    print(kk, vv.shape if isinstance(vv, torch.Tensor) else vv)
        else:
            print(k, v)
