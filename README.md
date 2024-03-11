# emager-py

[![pdm-managed](https://img.shields.io/badge/pdm-managed-blueviolet)](https://pdm-project.org)

This repo is a library to interact with EMaGer HD-sEMG cuff as well as the EMaGer Dataset. Feel free to contribute to it!

## Setting up

### Redis

Some modules in this library depend on [_Redis_](https://redis.io/), a fast and easy-to-use in-memory DB. Notably, it is used for extremely easy client/server interactions between a remote device (usually running sampling/on-board ML) and a host workstation for visualization, model training, etc.

### PDM

The repo is made with [PDM](https://pdm-project.org/latest/) in mind. It allows to easily create, develop, install and publish Python projects. It also handles dependency management. Let's say you want to create `my_test_project` and use `emager-py`.

First, you'd create the project directory:

```bash
mkdir my_test_project
cd my_test_project
```

Then, install PDM and run:

```bash
pdm venv create
pdm use .venv/bin/python
eval $(pdm venv activate)
pdm init # now answer the questions
# Track GitHub repo
pdm add -e "git+https://github.com/SBIOML/emager-py@main" --dev
# OR track local repo
pdm add -e file:///path/to/emager-py --dev
```

To update the dependencies: `pdm sync`

### Pip

You can install `emager-py` with `pip`:`

- From local: `python3 -m pip install -e file:///path/to/emager-py`
- From remote: `python3 -m pip install -e "git+https://github.com/SBIOML/emager-py@main#egg=emager-py"`

## Repository structure

- `emager_dataset` contains utilities to load, inspect and export EMaGer data. It is usually the entry point to `emager-py`
- `data_processing` contains data preprocessing utilities. For example, signal preprocessing, ABSDA data augmentation and label extraction
- `quantization` contains data quantization routines, which are especially useful for tinyML applications
- `transforms` are end-to-end data processing functions which take in raw EMaGer data and output ready-for-inference data. Usually, they take care of preprocessing and quantization if needed
- `emager_data_generator` is used in online experiments to emulate a live sampling process. It pushes data into a Redis database
- `emager_utils` is just general purpose constants and utilities
- `visualize` some tools to visualize signals
- `finn/` contains FINN-specific routines

## EMaGer Dataset

Refer to _emager_ repo to know more about the EMaGer dataset.
