#!/bin/bash

# Download datasets
python benchmarks/download_datasets.py

# Run evolution with specified number of generations and population size
python evolution/evolve.py --generations 20 --population 30
