import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

from ..utils.logging import Logger

class BenchmarkPlotter:
    """Visualization module for benchmark results."""
    
    def __init__(
        self,
        results: Dict,
        output_dir: Optional[Path] = None,
        logger: Optional[Logger] = None
    ):
        """Initialize plotter with benchmark results."""
        self.results = results
        self.output_dir = output_dir or Path("results/plots")
        self.logger = logger or Logger("plotter")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.style.use("seaborn")
        sns.set_palette("husl")
    
    def _save_plot(self, fig: plt.Figure, filename: str):
        """Save plot to file."""
        output_path = self.output_dir / filename
        fig.savefig(output_path, bbox_inches="tight", dpi=300)
        self.logger.info(f"Saved plot to {output_path}")
    
    def plot_latency_comparison(self):
        """Plot latency comparison between backends and precisions."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        x = []
        y = []
        hue = []
        for key, result in self.results.items():
            backend, precision, batch_size = key.split("_")
            x.append(int(batch_size))
            y.append(result["latency"]["mean"])
            hue.append(f"{backend}_{precision}")
        
        # Create plot
        sns.barplot(x=x, y=y, hue=hue, ax=ax)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Comparison by Backend and Precision")
        plt.xticks(rotation=45)
        
        self._save_plot(fig, "latency_comparison.png")
    
    def plot_throughput_comparison(self):
        """Plot throughput comparison between backends and precisions."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        x = []
        y = []
        hue = []
        for key, result in self.results.items():
            backend, precision, batch_size = key.split("_")
            x.append(int(batch_size))
            y.append(result["throughput"]["mean"])
            hue.append(f"{backend}_{precision}")
        
        # Create plot
        sns.barplot(x=x, y=y, hue=hue, ax=ax)
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput Comparison by Backend and Precision")
        plt.xticks(rotation=45)
        
        self._save_plot(fig, "throughput_comparison.png")
    
    def plot_gpu_metrics(self):
        """Plot GPU metrics for each configuration."""
        metrics = ["utilization", "memory", "power", "temperature"]
        for metric in metrics:
            if any(metric in result["gpu"] for result in self.results.values()):
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Prepare data
                x = []
                y = []
                hue = []
                for key, result in self.results.items():
                    if metric in result["gpu"]:
                        backend, precision, batch_size = key.split("_")
                        x.append(int(batch_size))
                        y.append(result["gpu"][metric]["mean"])
                        hue.append(f"{backend}_{precision}")
                
                # Create plot
                sns.barplot(x=x, y=y, hue=hue, ax=ax)
                ax.set_xlabel("Batch Size")
                ax.set_ylabel(f"GPU {metric.capitalize()}")
                ax.set_title(f"GPU {metric.capitalize()} by Backend and Precision")
                plt.xticks(rotation=45)
                
                self._save_plot(fig, f"gpu_{metric}.png")
    
    def plot_latency_distribution(self):
        """Plot latency distribution for each configuration."""
        fig, axes = plt.subplots(
            nrows=len(self.results),
            figsize=(12, 4 * len(self.results))
        )
        
        for (key, result), ax in zip(self.results.items(), axes):
            backend, precision, batch_size = key.split("_")
            
            # Create violin plot
            sns.violinplot(
                y=result["latency"]["values"],
                ax=ax
            )
            
            ax.set_title(
                f"Latency Distribution - {backend} {precision} (Batch Size: {batch_size})"
            )
            ax.set_ylabel("Latency (ms)")
        
        plt.tight_layout()
        self._save_plot(fig, "latency_distribution.png")
    
    def plot_throughput_vs_latency(self):
        """Plot throughput vs latency trade-off."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        x = []
        y = []
        hue = []
        for key, result in self.results.items():
            backend, precision, batch_size = key.split("_")
            x.append(result["latency"]["mean"])
            y.append(result["throughput"]["mean"])
            hue.append(f"{backend}_{precision}")
        
        # Create scatter plot
        sns.scatterplot(x=x, y=y, hue=hue, s=100, ax=ax)
        
        # Add batch size labels
        for key, result in self.results.items():
            backend, precision, batch_size = key.split("_")
            ax.annotate(
                batch_size,
                (result["latency"]["mean"], result["throughput"]["mean"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center"
            )
        
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput vs Latency Trade-off")
        
        self._save_plot(fig, "throughput_vs_latency.png")
    
    def generate_all_plots(self):
        """Generate all available plots."""
        self.plot_latency_comparison()
        self.plot_throughput_comparison()
        self.plot_gpu_metrics()
        self.plot_latency_distribution()
        self.plot_throughput_vs_latency() 