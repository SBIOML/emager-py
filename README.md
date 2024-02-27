# emager-py

This repository is meant to be submoduled in other projects that interact with EMaGer dataset. Feel free to contribute to it!

## Repository structure

- `emager_dataset` contains utilities to load, inspect and export EMaGer data. It is usually the entry point to `emager-py`
- `data_processing` contains data preprocessing utilities. For example, signal preprocessing, ABSDA data augmentation and label extraction
- `quantization` contains data quantization routines, which are especially useful for tinyML applications
- `transforms` are end-to-end data processing functions which take in raw EMaGer data and output ready-for-inference data. Usually, they take care of preprocessing and quantization if needed
- `emager_data_generator` is used in online experiments to emulate a live sampling process. It pushes data into a Redis database
- `emager_utils` is just general purpose constants and utilities
- `finn/` contains FINN-specific routines, used in `emager-torch`

## EMaGer Dataset

Refer to _emager_ repo to know more about the EMaGer dataset.
