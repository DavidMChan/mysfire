# Mysfire - Load data faster than light :)

Mysfire takes the headache out of writing dataset and data loader code for pytorch (that you usually repeat time and
time again). Mysfire encourages code reuse between projects when possible, and allows for easy extensibility when code
reuse is impossible. Not only this, mysfire makes it easy to scale your datasets to hundreds of nodes, without thinking:
cloud storage support is built in (and easy to extend), making it a powerful tool when going from your local laptop to
your public or private cloud.

## Installation

Install this library with pip - `pip install mysfire[all]`

For a restricted subset of the data loading types, you can use different options:

```bash
pip install mysfire # Default options, only basic processors
pip install mysfire[s3] # Include options for S3 connection
pip install mysfire[image] # Include image processors
pip install mysfire[video] # Include video processors
pip install mysfire[h5py] # Include H5py processors
pip install mysfire[nlp] # Include NLP processors
```

## Tour

Each mysfire dataset is composed of three components:

1. A definition describing the types of data (and preprocessing steps) in each column of your tabular file. Usually,
   this is just the header of your CSV or TSV file.
2. A tabular data store (usually just a CSV or TSV file, but we can load tabular data from S3, SQL or any other
   extensible columnular store)
3. A set of processors for processing and loading the data. For most common data types, these processors are built in,
   but we recognize that every dataset is different, so we make it as easy as possible to add new processors, or
   download third party processors from the mysfire community hub.

Let's look at a hello-world mysfire dataset:

```tsv
# simple_dataset.tsv
class:int   data:npy
0   sample_0.npy
1   sample_1.npy
2   sample_2.npy
```

That's it. Easy as defining the types of each of the objects and a name for each column as a header in a TSV file. The
data is then super easy to load to your normal PyTorch workflow:

```py
from mysfire import DataLoader
# Returns a standard PyTorch DataLoader, just replace the dataset with the TSV file!
train_dataloader = DataLoader('simple_dataset.tsv', batch_size = 3, num_workers=12)
for batch in train_dataloader:
    print(batch)
```

This dataset will produce a dictionary:

```py
{
    'class': [0, 1, 2]
    'data': np.ndarray # Array of shape [BS x ...]
}
```

We handle loading, collating, and batching the data, so you can focus on training models, and iterating on experiments.
Onboarding new datasets is as easy as setting up the new TSV file, and changing the links. No more messing around with
the code to add a new dataset switch! No coding that numpy loading dataset for the 100th time either - we've already
learned to handle all kinds of numpy types (even ragged arrays!)

Need S3? That's as easy as configuring a column with your S3 details:

```
# simple_s3_dataset.tsv
class:int   data:npy(s3_access_key="XXX",s3_secret_key="XXX",s3_endpoint="XXX")
0   s3://data/sample_0.npy
1   s3://data/sample_1.npy
2   s3://data/sample_2.npy
```

Merging two S3 sources? Configure each column independently:

```
# multisource_s3_dataset.tsv
class:int  data_a:npy(s3_access_key="AAA",s3_secret_key="AAA",s3_endpoint="AAA")    data_b:npy(s3_access_key="BBB",s3_secret_key="BBB",s3_endpoint="BBB")
0   s3://data/sample_0.npy   s3://data/sample_0.npy
1   s3://data/sample_1.npy   s3://data/sample_1.npy
2   s3://data/sample_2.npy   s3://data/sample_2.npy
```

Worried about putting your keys in a dataset file? Use `$S3_SECRET_KEY` (a `$` prefix) to load environment variables at
runtime.

```
# simple_s3_dataset.tsv
class:int   data:npy(s3_access_key=$S3_ACCESS_KEY,s3_secret_key=$S3_SECRET_KEY,s3_endpoint=$S3_ENDPOINT)
0   s3://data/sample_0.npy
1   s3://data/sample_1.npy
2   s3://data/sample_2.npy
```

Loading images or video?

```
# multimedia_s3_dataset.tsv
class:int   picture:img(resize=256)  frames:video(uniform_temporal_subsample=16)
0   image_1.png     video_1.mp4
1   image_2.jpg     video_2.mp4
2   image_3.JPEG     video_3.mp4
```

Need to do NLP? Huggingface Tokenizers is built in

```
# tokenization_s3_dataset.tsv
class:int   labels:nlp.huggingface_tokenization(tokenizer_json="./tokenizer.json")
0   Hello world!
1   Welcome to the Mysfire data processors
```

Working with PyTorch Lightning? LightningDataModules are built in:

```py
from mysfire import LightningDataModule
datamodule = LightningDataModule(
    'train.tsv',
    'validate.tsv',
    'test.tsv'
)
```

Need to run something at test-time? All you need to do is build a OneShotLoader:

```py
from mysfire import OneShotLoader

loader = OneShotLoader(filename='train.tsv') # Initialize from a TSV
loader = OneShotLoader(columns=["class:int", "data:npy"]) # or pass the columns directly!


data = loader([["field 1", "field 2"],["field 1", "field 2"]]) # Load data with a single method
```

Need to load a custom datatype? Or extend the existing datatypes? It's super easy:

```py
from mysfire import register_processor, Processor

# Register the processor with mysfire before creating a dataset
@register_processor
class StringAppendProcessor(Processor):

    # Setup an init function with any optional arguments that are parsed from the column. We handle all of the
    # complicated parsing for you, just take all options as Optional[str] arguments!
    def __init__(self, string_to_append: Optional[str] = None):
        self._string_to_append = string_to_append

    # Define a typestring that is matched against the TSV columns. Registered processors take precidence over
    # processors that are loaded by default
    @classmethod
    def typestr(cls):
        return "str"

    # Define a collate function for your data type which handles batching. If this is missing, we use the standard
    # torch collate function instead
    def collate(self, batch: List[Optional[str]]) -> List[str]:
        return [b or "" for b in batch]

    # Add a call function which transforms the string data in the TSV into a single data sample.
    def __call__(self, value: str) -> str:
        return value + self._string_to_append if self._string_to_append else ""
```

Want to add remote data loading to your processor? It's as easy as:

```py
from mysfire import register_processor, S3Processor

# Start by extending the S3 processor
@register_processor
class S3FileProcessor(S3Processor):
    def __init__(self,
                 s3_endpoint: Optional[str] = None,
                 s3_access_key: Optional[str] = None,
                 s3_secret_key: Optional[str] = None,
                 s3_region: Optional[str] = None,):

        super().__init__(
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_region=s3_region,
        )

    @classmethod
    def typestr(cls):
        return "str"

    def collate(self, batch: List[Optional[str]]) -> List[str]:
        return [b or "" for b in batch]

    def __call__(self, value: str) -> Optional[str]:
        try:
            # Use resolve_to_local to fetch any file in S3 to a local filepath (or use a local file path if it's local)
            with self.resolve_to_local(value) as f:
                with open(f, 'r') as fp:
                    return f
        except Exception as ex:
            return None
```

For full details, and to check out everything that we offer, check out our docs!

## Useful?

Cite us!

```
Bibtex
```
