# Warehouse.ML

This is a series of tools and pipelines to extract and compute probalistic data from images of pallets.

## Installation

You need to install anaconda for the package management first.


### Execute whole pipeline

This consists of:

1. indexing files
1. segmenting pallet front based on annotations
1. extracting muliple features from the segments: edges, corners, template matching, shadows under the pallet
1. Computing a probability distributions amond each category of extracted feature sets

```bash
python .\src\main.py pipeline .\data\data5 .\data\templates cam0
```
