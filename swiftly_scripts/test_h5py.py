import torch

from swiftly import Dataset

ds = Dataset("/home/davidchan/Projects/swiftly/swiftly_scripts/test_h5py.tsv")
dataloader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ds.collate_fn)
for batch in dataloader:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                print(kk, vv.shape)
        else:
            print(k, v)
