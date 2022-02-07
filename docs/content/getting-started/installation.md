---
title: "Installation"
date: 2022-02-04T14:37:49-05:00
draft: false
---

## Installation

This library is best installed using pip, with the following command: `pip install mysfire[all]`

Mysfire relies on a large library of processors to handle each of the different datatypes. The "[all]" option is
very permissive, and installs a lare number of packages for handling almost any data type that can come your way. If you
want to be somewhat more restrictive of the packages that your downstream library relies on, you can use the following
options:

```bash
pip install mysfire # Default options, only basic processors
pip install mysfire[s3] # Include options for S3 connection
pip install mysfire[image] # Include image processors
pip install mysfire[video] # Include video processors
pip install mysfire[h5py] # Include H5py processors
pip install mysfire[nlp] # Include NLP processors
```
