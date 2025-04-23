import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from ..utils.logging import Logger
from ..config import ModelConfig

@dataclass
class PyTorchModel:
    """Container for PyTorch model and tokenizer."""
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    device: str
    precision: str
    config: ModelConfig

class PyTorchBackend:
    """Enhanced PyTorch backend with support for different precisions and optimizations."""
    
    def __init__(self, config: ModelConfig, logger: Optional[Logger] = None):
        """Initialize PyTorch backend."""
        self.config = config
        self.logger = logger or Logger("pytorch_backend")
        self.models: Dict[str, PyTorchModel] = {}
    
    def _get_dtype(self, precision: str) -> torch.dtype:
        """Get PyTorch dtype from precision string."""
        if precision == "fp32":
            return torch.float32
        elif precision == "fp16":
            return torch.float16
        elif precision == "bf16":
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported precision: {precision}")
    
    def _setup_quantization(self, model: torch.nn.Module, precision: str) -> torch.nn.Module:
        """Setup model quantization if requested."""
        if precision == "int8":
            self.logger.info("Applying INT8 quantization")
            from torch.quantization import quantize_dynamic
            model = quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        elif precision == "int4":
            self.logger.info("Applying INT4 quantization")
            # Note: INT4 quantization requires custom implementation
            # This is a placeholder for future implementation
            raise NotImplementedError("INT4 quantization not yet implemented")
        return model
    
    def load_model(
        self,
        model_name: str,
        device: str = "cuda",
        precision: str = "fp16",
        use_cache: bool = True
    ) -> PyTorchModel:
        """Load a model with specified precision and optimizations."""
        cache_key = f"{model_name}_{device}_{precision}"
        
        if cache_key in self.models and use_cache:
            self.logger.info(f"Using cached model: {cache_key}")
            return self.models[cache_key]
        
        self.logger.info(f"Loading PyTorch model: {model_name} with {precision} precision")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with specified precision
        dtype = self._get_dtype(precision)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            trust_remote_code=self.config.trust_remote_code
        )
        
        # Apply quantization if needed
        if precision in ["int8", "int4"]:
            model = self._setup_quantization(model, precision)
        
        # Move model to device if not using device_map
        if device != "auto":
            model = model.to(device)
        
        # Apply optimizations
        model = self._optimize_model(model)
        
        pytorch_model = PyTorchModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            precision=precision,
            config=self.config
        )
        
        if use_cache:
            self.models[cache_key] = pytorch_model
        
        return pytorch_model
    
    def _optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply model optimizations."""
        self.logger.info("Applying model optimizations")
        
        # Enable CUDA graphs if available
        if torch.cuda.is_available():
            model = torch.compile(model, mode="reduce-overhead")
        
        # Enable model optimizations
        model.eval()
        with torch.no_grad():
            model = torch.jit.script(model)
        
        return model
    
    def generate(
        self,
        model: PyTorchModel,
        prompt: str,
        max_length: int = 128,
        batch_size: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> Tuple[torch.Tensor, float]:
        """Generate text with the model and return tokens and latency."""
        import time
        
        # Tokenize input
        inputs = model.tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Run generation
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=model.tokenizer.eos_token_id,
                do_sample=temperature > 0
            )
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        return outputs, latency
    
    def get_model_size(self, model: PyTorchModel) -> int:
        """Get model size in bytes."""
        param_size = 0
        for param in model.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
    
    def get_memory_usage(self, model: PyTorchModel) -> Dict[str, int]:
        """Get current memory usage."""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
                "max_reserved": torch.cuda.max_memory_reserved()
            }
        return {} 