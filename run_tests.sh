#!/bin/bash

# Install dependencies
pip install -r requirements.txt
pip install pytest

# Run tests
pytest tests/ -v

# Clean up test artifacts
rm -rf test_results/
rm -f test_config.yaml 