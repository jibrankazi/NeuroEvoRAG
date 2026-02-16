#!/bin/bash

echo "Downloading datasets..."
python benchmarks/download_datasets.py

echo "Running evolution..."
python evolution/evolve.py --generations 20 --population 30
