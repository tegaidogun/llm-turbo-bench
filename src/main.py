import argparse
from pathlib import Path
import yaml

from config import Config
from metrics.benchmark import BenchmarkRunner
from visualization.plotter import BenchmarkPlotter
from utils.logging import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Turbo Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to benchmark (overrides config)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "tensorrt", "both"],
        help="Backend to use (overrides config)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        help="Batch sizes to test (overrides config)"
    )
    parser.add_argument(
        "--precisions",
        type=str,
        nargs="+",
        choices=["fp32", "fp16", "int8", "int4"],
        help="Precisions to test (overrides config)"
    )
    return parser.parse_args()

def main():
    """Main entry point for the benchmark."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.model:
        config.benchmark.model_name = args.model
    if args.backend:
        config.benchmark.backend = args.backend
    if args.batch_sizes:
        config.benchmark.batch_sizes = args.batch_sizes
    if args.precisions:
        config.benchmark.precision = args.precisions[0]  # Use first precision as default
    
    # Setup logging
    logger = setup_logging(config.logging)
    logger.info("Starting LLM Turbo Benchmark")
    
    try:
        # Initialize benchmark runner
        runner = BenchmarkRunner(
            benchmark_config=config.benchmark,
            model_config=config.model,
            logger=logger
        )
        
        # Run benchmarks
        logger.info("Running benchmarks...")
        results = runner.run_benchmark(
            backend=config.benchmark.backend,
            batch_sizes=config.benchmark.batch_sizes,
            precisions=args.precisions or [config.benchmark.precision]
        )
        
        # Analyze results
        logger.info("Analyzing results...")
        analysis = runner.analyze_results(results)
        
        # Save results
        if config.benchmark.save_raw_data:
            output_dir = Path(config.benchmark.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw results
            with open(output_dir / "raw_results.yaml", "w") as f:
                yaml.dump(
                    {k: v.__dict__ for k, v in results.items()},
                    f,
                    default_flow_style=False
                )
            
            # Save analysis
            with open(output_dir / "analysis.yaml", "w") as f:
                yaml.dump(analysis, f, default_flow_style=False)
        
        # Generate plots
        if config.benchmark.generate_plots:
            logger.info("Generating plots...")
            plotter = BenchmarkPlotter(
                results=analysis,
                output_dir=Path(config.benchmark.output_dir) / "plots",
                logger=logger
            )
            plotter.generate_all_plots()
        
        logger.info("Benchmark completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main() 