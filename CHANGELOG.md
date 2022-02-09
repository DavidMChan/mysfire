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
