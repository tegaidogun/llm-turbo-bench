import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from ..backends.pytorch import PyTorchBackend, PyTorchModel
from ..backends.tensorrt import TensorRTBackend, TensorRTModel
from ..metrics.gpu_metrics import GPUMonitor
from ..utils.logging import Logger
from ..config import BenchmarkConfig, ModelConfig

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    latencies: List[float]
    throughputs: List[float]
    gpu_metrics: List[Dict]
    batch_size: int
    precision: str
    backend: str
    model_name: str

class BenchmarkRunner:
    """Enhanced benchmark runner with support for batch size analysis and advanced metrics."""
    
    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
        model_config: ModelConfig,
        logger: Optional[Logger] = None
    ):
        """Initialize benchmark runner."""
        self.benchmark_config = benchmark_config
        self.model_config = model_config
        self.logger = logger or Logger("benchmark_runner")
        
        # Initialize backends
        self.pytorch_backend = PyTorchBackend(model_config, logger)
        self.tensorrt_backend = TensorRTBackend(model_config, logger)
        
        # Initialize GPU monitor
        self.gpu_monitor = GPUMonitor(model_config, logger)
    
    def _warmup(
        self,
        model: PyTorchModel | TensorRTModel,
        prompt: str,
        max_length: int,
        batch_size: int,
        num_runs: int = 3
    ):
        """Run warmup iterations."""
        self.logger.info(f"Running {num_runs} warmup iterations")
        for _ in range(num_runs):
            if isinstance(model, PyTorchModel):
                self.pytorch_backend.generate(
                    model, prompt, max_length, batch_size
                )
            else:
                self.tensorrt_backend.generate(
                    model, prompt, max_length, batch_size
                )
    
    def _run_benchmark(
        self,
        model: PyTorchModel | TensorRTModel,
        prompt: str,
        max_length: int,
        batch_size: int,
        num_runs: int
    ) -> BenchmarkResult:
        """Run benchmark for a single configuration."""
        # Initialize result containers
        latencies = []
        throughputs = []
        gpu_metrics = []
        
        # Start GPU monitoring
        self.gpu_monitor.start()
        
        # Run benchmark
        for _ in tqdm(range(num_runs), desc="Running benchmark"):
            # Run inference
            if isinstance(model, PyTorchModel):
                outputs, latency = self.pytorch_backend.generate(
                    model, prompt, max_length, batch_size
                )
                num_tokens = len(outputs[0]) - len(model.tokenizer.encode(prompt))
            else:
                outputs, latency = self.tensorrt_backend.generate(
                    model, prompt, max_length, batch_size
                )
                num_tokens = len(model.tokenizer.encode(outputs[0])) - len(model.tokenizer.encode(prompt))
            
            # Calculate metrics
            throughput = (num_tokens * batch_size) / (latency / 1000)  # tokens/sec
            
            # Store results
            latencies.append(latency)
            throughputs.append(throughput)
            gpu_metrics.append(self.gpu_monitor.get_current_metrics())
        
        # Stop GPU monitoring
        self.gpu_monitor.stop()
        
        return BenchmarkResult(
            latencies=latencies,
            throughputs=throughputs,
            gpu_metrics=gpu_metrics,
            batch_size=batch_size,
            precision=model.precision,
            backend="pytorch" if isinstance(model, PyTorchModel) else "tensorrt",
            model_name=self.benchmark_config.model_name
        )
    
    def run_benchmark(
        self,
        backend: str = "both",
        batch_sizes: Optional[List[int]] = None,
        precisions: Optional[List[str]] = None
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across different configurations."""
        results = {}
        
        # Use default values if not specified
        batch_sizes = batch_sizes or self.benchmark_config.batch_sizes
        precisions = precisions or [self.benchmark_config.precision]
        
        # Run benchmarks for each configuration
        for precision in precisions:
            self.logger.info(f"Running benchmarks with {precision} precision")
            
            # Load models
            if backend in ["pytorch", "both"]:
                pytorch_model = self.pytorch_backend.load_model(
                    self.benchmark_config.model_name,
                    device=self.benchmark_config.device,
                    precision=precision
                )
            
            if backend in ["tensorrt", "both"]:
                tensorrt_model = self.tensorrt_backend.load_model(
                    self.benchmark_config.model_name,
                    precision=precision
                )
            
            # Run benchmarks for each batch size
            for batch_size in batch_sizes:
                self.logger.info(f"Running benchmark with batch size {batch_size}")
                
                if backend in ["pytorch", "both"]:
                    # Warmup
                    self._warmup(
                        pytorch_model,
                        self.benchmark_config.prompt,
                        self.benchmark_config.max_length,
                        batch_size
                    )
                    
                    # Run benchmark
                    result = self._run_benchmark(
                        pytorch_model,
                        self.benchmark_config.prompt,
                        self.benchmark_config.max_length,
                        batch_size,
                        self.benchmark_config.num_runs
                    )
                    
                    key = f"pytorch_{precision}_{batch_size}"
                    results[key] = result
                
                if backend in ["tensorrt", "both"]:
                    # Warmup
                    self._warmup(
                        tensorrt_model,
                        self.benchmark_config.prompt,
                        self.benchmark_config.max_length,
                        batch_size
                    )
                    
                    # Run benchmark
                    result = self._run_benchmark(
                        tensorrt_model,
                        self.benchmark_config.prompt,
                        self.benchmark_config.max_length,
                        batch_size,
                        self.benchmark_config.num_runs
                    )
                    
                    key = f"tensorrt_{precision}_{batch_size}"
                    results[key] = result
        
        return results
    
    def analyze_results(self, results: Dict[str, BenchmarkResult]) -> Dict:
        """Analyze benchmark results and generate statistics."""
        analysis = {}
        
        for key, result in results.items():
            # Calculate basic statistics
            stats = {
                "latency": {
                    "mean": np.mean(result.latencies),
                    "std": np.std(result.latencies),
                    "min": np.min(result.latencies),
                    "max": np.max(result.latencies),
                    "p95": np.percentile(result.latencies, 95),
                    "p99": np.percentile(result.latencies, 99)
                },
                "throughput": {
                    "mean": np.mean(result.throughputs),
                    "std": np.std(result.throughputs),
                    "min": np.min(result.throughputs),
                    "max": np.max(result.throughputs)
                }
            }
            
            # Calculate GPU metrics
            gpu_stats = {
                "utilization": {
                    "mean": np.mean([m.utilization["gpu"] for m in result.gpu_metrics]),
                    "max": np.max([m.utilization["gpu"] for m in result.gpu_metrics])
                },
                "memory": {
                    "mean": np.mean([m.memory["used"] for m in result.gpu_metrics]),
                    "max": np.max([m.memory["used"] for m in result.gpu_metrics])
                }
            }
            
            # Add power and temperature if available
            power_readings = [m.power for m in result.gpu_metrics if m.power is not None]
            if power_readings:
                gpu_stats["power"] = {
                    "mean": np.mean(power_readings),
                    "max": np.max(power_readings)
                }
            
            temp_readings = [m.temperature for m in result.gpu_metrics if m.temperature is not None]
            if temp_readings:
                gpu_stats["temperature"] = {
                    "mean": np.mean(temp_readings),
                    "max": np.max(temp_readings)
                }
            
            stats["gpu"] = gpu_stats
            analysis[key] = stats
        
        return analysis 