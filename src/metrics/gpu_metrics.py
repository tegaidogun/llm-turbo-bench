import time
import threading
from typing import List, Dict, Optional
import pynvml
import psutil
from dataclasses import dataclass
from ..utils.logging import Logger

@dataclass
class GPUMetrics:
    """Container for GPU metrics."""
    utilization: Dict[str, int]  # GPU and memory utilization percentages
    memory: Dict[str, int]      # Memory usage in bytes
    power: Optional[float]      # Power usage in watts
    temperature: Optional[float] # Temperature in Celsius
    timestamp: float           # Unix timestamp

class GPUMonitor:
    """Enhanced GPU monitoring with power and temperature tracking."""
    
    def __init__(self, config, logger: Optional[Logger] = None):
        """Initialize GPU monitor with configuration."""
        self.config = config
        self.logger = logger or Logger("gpu_monitor")
        self.monitoring = False
        self.thread = None
        self.metrics_data: List[GPUMetrics] = []
        
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) 
                          for i in range(self.device_count)]
            self.logger.info(f"Initialized GPU monitoring for {self.device_count} GPUs")
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU monitoring: {e}")
            raise
    
    def _get_gpu_metrics(self) -> GPUMetrics:
        """Get current GPU metrics for all GPUs."""
        try:
            # Get utilization rates
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handles[0])
            utilization = {
                "gpu": util.gpu,
                "memory": util.memory
            }
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handles[0])
            memory = {
                "total": mem_info.total,
                "free": mem_info.free,
                "used": mem_info.used
            }
            
            # Get power usage if supported
            power = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handles[0]) / 1000.0  # Convert to watts
            except pynvml.NVMLError:
                pass
            
            # Get temperature if supported
            temperature = None
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    self.handles[0], pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                pass
            
            return GPUMetrics(
                utilization=utilization,
                memory=memory,
                power=power,
                temperature=temperature,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Error getting GPU metrics: {e}")
            raise
    
    def _monitor_loop(self):
        """Background thread to monitor GPU metrics."""
        while self.monitoring:
            try:
                metrics = self._get_gpu_metrics()
                self.metrics_data.append(metrics)
                time.sleep(self.config.sampling_interval)
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}")
                time.sleep(1)  # Prevent tight loop on errors
    
    def start(self):
        """Start monitoring GPU metrics in a background thread."""
        if not self.monitoring:
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.daemon = True
            self.thread.start()
            self.logger.info("Started GPU monitoring")
    
    def stop(self):
        """Stop monitoring GPU metrics."""
        self.monitoring = False
        if self.thread:
            self.thread.join()
            self.thread = None
        self.logger.info("Stopped GPU monitoring")
    
    def get_current_metrics(self) -> GPUMetrics:
        """Get the most recent GPU metrics."""
        return self._get_gpu_metrics()
    
    def get_average_metrics(self) -> Dict:
        """Get average metrics across all collected data."""
        if not self.metrics_data:
            return {}
        
        avg_metrics = {
            "gpu_utilization": sum(m.utilization["gpu"] for m in self.metrics_data) / len(self.metrics_data),
            "memory_utilization": sum(m.utilization["memory"] for m in self.metrics_data) / len(self.metrics_data),
            "memory_used": sum(m.memory["used"] for m in self.metrics_data) / len(self.metrics_data),
        }
        
        # Add power and temperature if available
        power_readings = [m.power for m in self.metrics_data if m.power is not None]
        if power_readings:
            avg_metrics["power"] = sum(power_readings) / len(power_readings)
        
        temp_readings = [m.temperature for m in self.metrics_data if m.temperature is not None]
        if temp_readings:
            avg_metrics["temperature"] = sum(temp_readings) / len(temp_readings)
        
        return avg_metrics
    
    def get_peak_metrics(self) -> Dict:
        """Get peak metrics across all collected data."""
        if not self.metrics_data:
            return {}
        
        peak_metrics = {
            "gpu_utilization": max(m.utilization["gpu"] for m in self.metrics_data),
            "memory_utilization": max(m.utilization["memory"] for m in self.metrics_data),
            "memory_used": max(m.memory["used"] for m in self.metrics_data),
        }
        
        # Add power and temperature if available
        power_readings = [m.power for m in self.metrics_data if m.power is not None]
        if power_readings:
            peak_metrics["power"] = max(power_readings)
        
        temp_readings = [m.temperature for m in self.metrics_data if m.temperature is not None]
        if temp_readings:
            peak_metrics["temperature"] = max(temp_readings)
        
        return peak_metrics
    
    def __del__(self):
        """Clean up NVML resources."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass 