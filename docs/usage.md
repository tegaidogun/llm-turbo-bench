# Usage Guide

This guide provides detailed instructions on how to use the LLM Turbo Benchmark tool.

## Basic Usage

### Running with Default Settings

```bash
# Activate virtual environment first
source venv/bin/activate  # or conda activate llm-bench

# Run benchmark with default settings
python src/main.py
```

### Using Custom Configuration

1. Create a configuration file (e.g., `custom_config.yaml`)
2. Run with custom config:
```bash
python src/main.py --config custom_config.yaml
```

## Command Line Arguments

### Basic Arguments

- `--config`: Path to configuration file
  ```bash
  python src/main.py --config configs/my_config.yaml
  ```

- `--model`: Specify model name
  ```bash
  python src/main.py --model "facebook/opt-6.7b"
  ```

- `--backend`: Choose backend (`pytorch`, `tensorrt`, or `both`)
  ```bash
  python src/main.py --backend tensorrt
  ```

### Advanced Arguments

- `--batch-sizes`: Test specific batch sizes
  ```bash
  python src/main.py --batch-sizes 1 2 4 8
  ```

- `--precisions`: Test specific precisions
  ```bash
  python src/main.py --precisions fp16 int8
  ```

- `--num-runs`: Number of benchmark runs
  ```bash
  python src/main.py --num-runs 10
  ```

- `--output-dir`: Custom output directory
  ```bash
  python src/main.py --output-dir results/my_benchmark
  ```

## Configuration Examples

### Basic Configuration

```yaml
model_name: "facebook/opt-6.7b"
backend: "pytorch"
num_runs: 5
batch_sizes: [1, 2, 4]
precision: "fp16"
```

### Advanced Configuration

```yaml
model_name: "facebook/opt-6.7b"
backend: "both"
num_runs: 10
batch_sizes: [1, 2, 4, 8, 16]
precisions: ["fp16", "int8"]
device: "cuda"
warmup_runs: 3
gpu_monitor_interval: 0.1
output_dir: "results/advanced_benchmark"
```

## Jupyter Notebook Usage

### Analyzing Results

1. Launch Jupyter:
```bash
jupyter notebook notebooks/
```

2. Open `analyze_results.ipynb`
3. Load your benchmark results
4. Use the interactive widgets to analyze data

### Interactive Benchmarking

1. Open `interactive_benchmark.ipynb`
2. Configure benchmark settings using widgets
3. Run benchmarks and view results in real-time

## Best Practices

1. **Warmup Runs**
   - Always include warmup runs in your configuration
   - Helps stabilize performance measurements

2. **Batch Size Selection**
   - Start with small batch sizes
   - Gradually increase to find optimal size
   - Consider GPU memory constraints

3. **Precision Selection**
   - Use FP16 for general benchmarking
   - Try INT8 for maximum performance
   - Consider accuracy requirements

4. **Result Analysis**
   - Check for outliers in measurements
   - Compare different configurations
   - Consider GPU utilization metrics 