# Installation Guide

This guide will help you set up the LLM Turbo Benchmark project using a virtual environment.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- CUDA-compatible GPU (for GPU acceleration)
- Git

## Setting Up Virtual Environment

### Using venv (Recommended)

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Linux/Mac:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Using conda

1. Create a conda environment:
```bash
conda create -n llm-bench python=3.8
```

2. Activate the environment:
```bash
conda activate llm-bench
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Verifying Installation

To verify your installation, run:
```bash
python src/main.py --help
```

You should see the command-line help output.

## Troubleshooting

### Common Issues

1. **CUDA not found**
   - Ensure CUDA is installed and properly configured
   - Check CUDA version compatibility with PyTorch

2. **Dependency conflicts**
   - Try creating a fresh virtual environment
   - Install dependencies one by one to identify conflicts

3. **Memory issues**
   - Reduce batch size in configuration
   - Use lower precision (FP16 or INT8)

### Getting Help

If you encounter issues:
1. Check the [FAQ](faq.md)
2. Search existing issues on GitHub
3. Create a new issue with detailed information about your setup and the problem 