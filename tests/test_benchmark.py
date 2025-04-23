import pytest
from pathlib import Path
import yaml

from src.config import Config
from src.metrics.benchmark import BenchmarkRunner
from src.visualization.plotter import BenchmarkPlotter
from src.utils.logging import setup_logging

@pytest.fixture
def config():
    """Create a test configuration."""
    return Config()

@pytest.fixture
def logger(config):
    """Create a test logger."""
    return setup_logging(config.logging)

def test_benchmark_runner_init(config, logger):
    """Test benchmark runner initialization."""
    runner = BenchmarkRunner(
        benchmark_config=config.benchmark,
        model_config=config.model,
        logger=logger
    )
    assert runner is not None
    assert runner.benchmark_config == config.benchmark
    assert runner.model_config == config.model

def test_benchmark_runner_warmup(config, logger):
    """Test benchmark warmup."""
    runner = BenchmarkRunner(
        benchmark_config=config.benchmark,
        model_config=config.model,
        logger=logger
    )
    
    # Test with small batch size to avoid memory issues
    config.benchmark.batch_sizes = [1]
    config.benchmark.num_runs = 2
    
    results = runner.run_benchmark(
        backend="pytorch",
        batch_sizes=[1],
        precisions=["fp16"]
    )
    
    assert len(results) > 0
    for key, result in results.items():
        assert len(result.latencies) == config.benchmark.num_runs
        assert len(result.throughputs) == config.benchmark.num_runs
        assert len(result.gpu_metrics) == config.benchmark.num_runs

def test_benchmark_analysis(config, logger):
    """Test benchmark analysis."""
    runner = BenchmarkRunner(
        benchmark_config=config.benchmark,
        model_config=config.model,
        logger=logger
    )
    
    # Run benchmark with minimal settings
    config.benchmark.batch_sizes = [1]
    config.benchmark.num_runs = 2
    
    results = runner.run_benchmark(
        backend="pytorch",
        batch_sizes=[1],
        precisions=["fp16"]
    )
    
    analysis = runner.analyze_results(results)
    
    assert len(analysis) > 0
    for key, stats in analysis.items():
        assert "latency" in stats
        assert "throughput" in stats
        assert "gpu" in stats
        
        latency = stats["latency"]
        assert "mean" in latency
        assert "std" in latency
        assert "min" in latency
        assert "max" in latency
        assert "p95" in latency
        assert "p99" in latency

def test_plotter(config, logger):
    """Test plotter functionality."""
    # Create test data
    test_data = {
        "pytorch_fp16_1": {
            "latency": {
                "mean": 100,
                "values": [90, 100, 110]
            },
            "throughput": {
                "mean": 10
            },
            "gpu": {
                "utilization": {"mean": 50},
                "memory": {"mean": 1024}
            }
        }
    }
    
    plotter = BenchmarkPlotter(
        results=test_data,
        output_dir=Path("test_results/plots"),
        logger=logger
    )
    
    # Test individual plots
    plotter.plot_latency_comparison()
    plotter.plot_throughput_comparison()
    plotter.plot_gpu_metrics()
    plotter.plot_latency_distribution()
    plotter.plot_throughput_vs_latency()
    
    # Test generating all plots
    plotter.generate_all_plots()
    
    # Verify plots were created
    output_dir = Path("test_results/plots")
    assert (output_dir / "latency_comparison.png").exists()
    assert (output_dir / "throughput_comparison.png").exists()
    assert (output_dir / "gpu_utilization.png").exists()
    assert (output_dir / "gpu_memory.png").exists()
    assert (output_dir / "latency_distribution.png").exists()
    assert (output_dir / "throughput_vs_latency.png").exists()

def test_config_loading():
    """Test configuration loading."""
    # Create test config file
    test_config = {
        "benchmark": {
            "model_name": "test_model",
            "backend": "pytorch",
            "num_runs": 5
        }
    }
    
    config_path = Path("test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    
    # Load config
    config = Config.from_yaml(config_path)
    
    assert config.benchmark.model_name == "test_model"
    assert config.benchmark.backend == "pytorch"
    assert config.benchmark.num_runs == 5
    
    # Clean up
    config_path.unlink() 