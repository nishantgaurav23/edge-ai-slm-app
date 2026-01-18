"""
Quantization Service - Dynamic Model Selection for Edge AI SLM App

Provides:
- Auto-selection of model quantization based on device RAM
- 4-bit (Q4_K_M) for devices with < 6GB RAM
- 8-bit (Q8_0) for devices with >= 6GB RAM
"""

import logging
from typing import Optional, Tuple
from app.services.hardware_monitor import HardwareMonitor

logger = logging.getLogger(__name__)


class QuantizationService:
    """
    Dynamically selects optimal model quantization based on device capabilities.
    """
    
    # Model quantization options
    QUANT_4BIT = "Q4_K_M"  # ~4.7 bits per weight, good quality/size balance
    QUANT_8BIT = "Q8_0"   # 8 bits per weight, higher quality
    
    # RAM thresholds
    RAM_THRESHOLD_GB = 6.0  # Devices with >= 6GB use 8-bit
    
    # Context size recommendations based on RAM
    CONTEXT_SIZES = {
        'low': 1024,     # < 4GB RAM
        'medium': 2048,  # 4-8GB RAM
        'high': 4096     # > 8GB RAM
    }
    
    @classmethod
    def get_optimal_quantization(cls) -> str:
        """
        Determine optimal quantization based on available RAM.
        
        Returns:
            str: Quantization suffix (e.g., 'Q4_K_M' or 'Q8_0')
        """
        ram_gb = HardwareMonitor.get_total_ram_gb()
        
        if ram_gb < cls.RAM_THRESHOLD_GB:
            logger.info(f"Device RAM: {ram_gb:.1f}GB - Using 4-bit quantization ({cls.QUANT_4BIT})")
            return cls.QUANT_4BIT
        else:
            logger.info(f"Device RAM: {ram_gb:.1f}GB - Using 8-bit quantization ({cls.QUANT_8BIT})")
            return cls.QUANT_8BIT
    
    @classmethod
    def get_optimal_context_size(cls) -> int:
        """
        Determine optimal context window size based on RAM.
        
        Returns:
            int: Recommended context size in tokens
        """
        ram_gb = HardwareMonitor.get_total_ram_gb()
        
        if ram_gb < 4.0:
            return cls.CONTEXT_SIZES['low']
        elif ram_gb < 8.0:
            return cls.CONTEXT_SIZES['medium']
        else:
            return cls.CONTEXT_SIZES['high']
    
    @classmethod
    def get_model_config(cls) -> dict:
        """
        Get complete model configuration based on device capabilities.
        
        Returns:
            dict: Configuration for model loading
        """
        ram_gb = HardwareMonitor.get_total_ram_gb()
        is_low_end = HardwareMonitor.is_low_end_device()
        
        config = {
            'quantization': cls.get_optimal_quantization(),
            'context_size': cls.get_optimal_context_size(),
            'n_gpu_layers': 0,  # CPU-only for mobile (adjust for Metal/GPU support)
            'n_threads': 4 if is_low_end else 8,
            'use_mlock': not is_low_end,  # Lock memory on higher-end devices
            'n_batch': 256 if is_low_end else 512,
        }
        
        logger.info(f"Model config for device ({ram_gb:.1f}GB RAM): {config}")
        return config
    
    @classmethod
    def get_recommended_model_filename(cls, base_name: str = "tinyllama") -> str:
        """
        Get recommended model filename based on quantization.
        
        Args:
            base_name: Base model name (e.g., 'tinyllama', 'phi-2')
            
        Returns:
            str: Full filename with quantization suffix
        """
        quant = cls.get_optimal_quantization()
        return f"{base_name}-{quant}.gguf"
    
    @classmethod
    def estimate_memory_usage(cls, model_size_gb: float, quantization: str) -> float:
        """
        Estimate memory usage for a model.
        
        Args:
            model_size_gb: Original model size in GB (full precision)
            quantization: Quantization type ('Q4_K_M', 'Q8_0', etc.)
            
        Returns:
            float: Estimated memory usage in GB
        """
        # Approximate compression ratios
        compression = {
            'Q4_K_M': 0.25,  # ~4x compression
            'Q8_0': 0.5,    # ~2x compression
            'Q4_0': 0.25,
            'Q5_K_M': 0.31,
        }
        
        ratio = compression.get(quantization, 0.5)
        # Add ~20% overhead for KV cache and runtime
        return model_size_gb * ratio * 1.2
    
    @classmethod
    def can_load_model(cls, model_memory_gb: float) -> Tuple[bool, str]:
        """
        Check if device can safely load a model.
        
        Args:
            model_memory_gb: Required memory for model
            
        Returns:
            Tuple of (can_load: bool, reason: str)
        """
        ram_gb = HardwareMonitor.get_total_ram_gb()
        
        # Reserve ~2GB for OS and other apps
        available_for_model = ram_gb - 2.0
        
        if model_memory_gb > available_for_model:
            return False, f"Model requires {model_memory_gb:.1f}GB but only {available_for_model:.1f}GB available"
        
        if model_memory_gb > available_for_model * 0.8:
            return True, f"Warning: Model will use {(model_memory_gb/available_for_model)*100:.0f}% of available memory"
        
        return True, "OK"
