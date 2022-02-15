## 0.3.0 (2022-02-15)

### Feat

- **mysfire**: Changed collate to flatten dictionaries at top level
- **mysfire/processors**: Adapted all processors to take compatible types directly
- **mysfire/processors/npy_processor.py**: Changed numpy processor to use boolean parsing
- **mysfire/processors**: Added pre_init function for processors to initialize global data-specific variables
- **mysfire/vars_registry.py**: Added ability to access properties from the dataset class

### Fix

- **mysfire/processors/video_processor.py**: Catch errors in video processor (return None, None)
- **mysfire/dataset.py**: Fixed issue with memory leaks in torch dataloader
- **mysfire/parser.py**: Fixed Reduce/Reduce conflict in grammar
- **mysfire/parser.py**: Added the ability to insert 1-element tuples in the header

### Perf

- **mysfire/processors/video_processor.py**: Added code to filter NaN values from audio data

## 0.2.0 (2022-02-09)

### Fix

- **mysfire/lightning**: Updated pytorch lightning dataloaders to only shuffle training data
- **mysfire/processors/video_processor.py**: Reworked fixed shape video processor to be CTHW vs. THWC
- **mysfire/parser.py**: Updated float regex so it doesn't match integer values
- **mysfire/processors/_processor.py**: Fixed import bug with S3 processor base type
- **setup.cfg**: Fixed issue with S3 not installing all requirements necessary
- **setup.cfg**: Fixed version issues preventing build

### Feat

- **mysfire/processors/nlp/tokenization_processor.py**: Added simple whitespace tokenizer
- **mysfire/processors/video_processor.py**: Added fixed-shape video processor
- **mysfire/parser.py**: Added support for BOOLs as possible options
- **mysfire/parser.py**: Add tuples as available parser type

### Refactor

- **mysfire/***: Improved mypy typing issues

## 0.1.2 (2022-01-26)

### Fix

- **setup.cfg**: Recursive import build issues

## 0.1.1 (2022-01-26)

## 0.1.0 (2022-01-26)

### Feat

- **README.md,-setup.cfg,-.pre-commit-config.yaml**: Added commitizen, CI, dev extras
