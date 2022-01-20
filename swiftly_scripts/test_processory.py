import torch

from swiftly import Dataset, register_processor_directory

register_processor_directory("/home/davidchan/Projects/swiftly/swiftly_scripts/extra_processors")

x = Dataset("/home/davidchan/Projects/swiftly/swiftly_scripts/simple_data.tsv")
dataloader = torch.utils.data.DataLoader(x, batch_size=2, shuffle=False, collate_fn=x.collate_fn)
for batch in dataloader:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
            # print(k, v)
        else:
            print(k, v)
