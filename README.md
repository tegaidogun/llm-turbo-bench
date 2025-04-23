# LLM Turbo Benchmark

A comprehensive benchmarking tool for evaluating LLM inference performance across different backends, precisions, and batch sizes.

## Features

- Support for PyTorch and TensorRT backends
- Multiple precision options (FP32, FP16, INT8, INT4)
- Batch size analysis
- GPU metrics monitoring
- Comprehensive visualization
- Configurable benchmarking parameters
- Interactive Jupyter notebooks for analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tegaidogun/llm-turbo-bench.git
cd llm-turbo-bench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the benchmark with default settings:
```bash
python src/main.py
```

Create a custom configuration file (e.g., `custom_config.yaml`) and specify it:
```bash
python src/main.py --config custom_config.yaml
```

### Jupyter Notebooks

The project includes two Jupyter notebooks for interactive analysis:

1. `notebooks/analyze_results.ipynb`: For analyzing existing benchmark results
   - Load and parse benchmark results
   - Generate interactive visualizations
   - Perform detailed analysis
   - Compare different configurations

2. `notebooks/interactive_benchmark.ipynb`: For running interactive benchmarks
   - Configure benchmarks using widgets
   - Run benchmarks with custom settings
   - View real-time results
   - Generate interactive plots

To use the notebooks:
```bash
jupyter notebook notebooks/
```

### Command Line Arguments

- `--config`: Path to configuration file (default: `config.yaml`)
- `--model`: Model name to benchmark (overrides config)
- `--backend`: Backend to use (`pytorch`, `tensorrt`, or `both`)
- `--batch-sizes`: Batch sizes to test (space-separated list)
- `--precisions`: Precisions to test (space-separated list: `fp32`, `fp16`, `int8`, `int4`)

Example:
```bash
python src/main.py --model "facebook/opt-6.7b" --backend both --batch-sizes 1 2 4 8 --precisions fp16 int8
```

## Configuration

The configuration file (`config.yaml`) supports the following sections:

### Benchmark Configuration
- `model_name`: Model to benchmark
- `backend`: Backend to use (`pytorch`, `tensorrt`, or `both`)
- `num_runs`: Number of benchmark runs
- `prompt`: Input prompt for benchmarking
- `max_length`: Maximum output length
- `batch_sizes`: List of batch sizes to test
- `precision`: Default precision
- `device`: Device to use (`cuda` or `cpu`)
- `warmup_runs`: Number of warmup runs
- `gpu_monitor_interval`: GPU metrics sampling interval
- `output_dir`: Output directory for results
- `save_raw_data`: Whether to save raw benchmark data
- `generate_plots`: Whether to generate plots

### Model Configuration
- `max_batch_size`: Maximum batch size
- `max_input_len`: Maximum input length
- `max_output_len`: Maximum output length
- `use_cache`: Whether to use model cache
- `trust_remote_code`: Whether to trust remote code
- `low_cpu_mem_usage`: Whether to use low CPU memory
- `quantization_config`: Quantization configuration

### Visualization Configuration
- `plot_style`: Plot style
- `figure_size`: Figure size
- `moving_average_window`: Moving average window
- `save_format`: Plot save format
- `dpi`: Plot DPI
- `interactive`: Whether to use interactive mode

### Monitoring Configuration
- `monitor_gpu`: Whether to monitor GPU
- `monitor_memory`: Whether to monitor memory
- `monitor_power`: Whether to monitor power
- `monitor_temperature`: Whether to monitor temperature
- `sampling_interval`: Sampling interval

### Logging Configuration
- `log_file`: Log file path
- `console_level`: Console log level
- `file_level`: File log level
- `format`: Log format
- `date_format`: Date format

## Results

The benchmark generates the following outputs:

1. Raw benchmark data in YAML format
2. Analysis results in YAML format
3. Various plots:
   - Latency comparison
   - Throughput comparison
   - GPU metrics
   - Latency distribution
   - Throughput vs latency trade-off

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 