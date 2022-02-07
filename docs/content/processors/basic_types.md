---
title: "Basic Types"
date: 2022-02-07T12:54:37-05:00
draft: false
---

## Summary


| Type String | Processor | Description | Example |
| ----------- | --------- | ----------- | ------- |
| int   | IntProcessor | Treats the column as a 32 bit integer | x:int |
| float   | FloatProcessor | Treats the column as a floating-point number | x:float |
| str   | StringProcessor | Treats the colum as an individual string | x:str |
| str.list   | StringListProcessor | Treats the colum as a list of strings, with a delimiter (default: ###) | x:str.list(delimiter="###") |




## Full API Docs


### mysfire.processors.base_types_processor.IntProcessor

Class-Level Docs

#### Functions

> mysfire.processors.base_types_processor.IntProcessor.\_\_init__(self, )

&nbsp;&nbsp;&nbsp;&nbsp; Function Level Docs

&nbsp;&nbsp;&nbsp;&nbsp;__Parameters__
: *self* - The initialized object

&nbsp;&nbsp;&nbsp;&nbsp;__Returns__
: None

> mysfire.processors.base_types_processor.IntProcessor.collate(self, )

&nbsp;&nbsp;&nbsp;&nbsp; Function Level Docs

&nbsp;&nbsp;&nbsp;&nbsp;__Parameters__
: *self* - The initialized object

&nbsp;&nbsp;&nbsp;&nbsp;__Returns__
: None

> mysfire.processors.base_types_processor.IntProcessor.\_\_call__(self, )

&nbsp;&nbsp;&nbsp;&nbsp; Function Level Docs

&nbsp;&nbsp;&nbsp;&nbsp;__Parameters__
: *self* - The initialized object

&nbsp;&nbsp;&nbsp;&nbsp;__Returns__
: None

#### Class Attributes

> mysfire.processors.base_types_processor.IntProcessor.typestr = 'int'
