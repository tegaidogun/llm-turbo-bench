from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model_name: str = "facebook/opt-6.7b"
    backend: str = "both"  # "pytorch", "trt", or "both"
    num_runs: int = 20
    prompt: str = "Hello, how are you?"
    max_length: int = 128
    batch_sizes: List[int] = (1, 2, 4, 8, 16)
    precision: str = "fp16"  # "fp32", "fp16", "int8", "int4"
    device: str = "cuda"
    warmup_runs: int = 3
    gpu_monitor_interval: float = 0.1
    output_dir: Path = Path("results")
    save_raw_data: bool = True
    generate_plots: bool = True
    log_level: str = "INFO"

@dataclass
class ModelConfig:
    """Configuration for model loading and optimization."""
    max_batch_size: int = 16
    max_input_len: int = 512
    max_output_len: int = 128
    use_cache: bool = True
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    quantization_config: Optional[dict] = None

@dataclass
class VisualizationConfig:
    """Configuration for result visualization."""
    plot_style: str = "seaborn"
    figure_size: tuple = (10, 6)
    moving_average_window: int = 5
    save_format: str = "png"  # "png", "pdf", "svg"
    dpi: int = 300
    interactive: bool = False

@dataclass
class MonitoringConfig:
    """Configuration for system monitoring."""
    monitor_gpu: bool = True
    monitor_memory: bool = True
    monitor_power: bool = False
    monitor_temperature: bool = True
    sampling_interval: float = 0.1

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_file: Optional[Path] = None
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

@dataclass
class Config:
    """Main configuration class combining all configs."""
    benchmark: BenchmarkConfig = BenchmarkConfig()
    model: ModelConfig = ModelConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Config':
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: Path):
        """Save configuration to YAML file."""
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False) 