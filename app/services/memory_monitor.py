"""
Memory Monitor Service - Memory Pressure Detection for Edge AI SLM App

Provides:
- Background monitoring of RAM usage
- Auto-unload models when memory pressure is high
- Preload models during idle time when memory is available
"""

import logging
import threading
import time
from typing import Callable, Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Monitors system memory and triggers actions based on pressure levels.
    """
    
    # Memory thresholds
    CRITICAL_THRESHOLD = 85  # Unload model above this %
    HIGH_THRESHOLD = 70      # Warning level
    LOW_THRESHOLD = 50       # Safe to preload
    
    # Monitoring interval in seconds
    CHECK_INTERVAL = 5.0
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._on_memory_critical: Optional[Callable] = None
        self._on_memory_available: Optional[Callable] = None
        self._last_state = 'normal'
    
    @staticmethod
    def get_memory_percent() -> float:
        """Get current memory usage percentage."""
        if psutil:
            return psutil.virtual_memory().percent
        return 50.0  # Default fallback
    
    @staticmethod
    def get_available_memory_gb() -> float:
        """Get available memory in GB."""
        if psutil:
            return psutil.virtual_memory().available / (1024 ** 3)
        return 2.0  # Default fallback
    
    def set_callbacks(
        self, 
        on_critical: Optional[Callable] = None,
        on_available: Optional[Callable] = None
    ):
        """
        Set callbacks for memory events.
        
        Args:
            on_critical: Called when memory exceeds critical threshold
            on_available: Called when memory becomes available (for preloading)
        """
        self._on_memory_critical = on_critical
        self._on_memory_available = on_available
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                self._check_memory_pressure()
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            
            time.sleep(self.CHECK_INTERVAL)
    
    def _check_memory_pressure(self):
        """Check memory pressure and trigger appropriate callbacks."""
        mem_percent = self.get_memory_percent()
        
        if mem_percent >= self.CRITICAL_THRESHOLD:
            if self._last_state != 'critical':
                logger.warning(f"Memory critical: {mem_percent:.1f}%")
                self._last_state = 'critical'
                if self._on_memory_critical:
                    self._on_memory_critical()
        
        elif mem_percent >= self.HIGH_THRESHOLD:
            if self._last_state != 'high':
                logger.info(f"Memory high: {mem_percent:.1f}%")
                self._last_state = 'high'
        
        elif mem_percent < self.LOW_THRESHOLD:
            if self._last_state != 'low':
                logger.info(f"Memory available: {mem_percent:.1f}%")
                self._last_state = 'low'
                if self._on_memory_available:
                    self._on_memory_available()
        
        else:
            self._last_state = 'normal'
    
    def get_memory_status(self) -> dict:
        """Get current memory status."""
        mem_percent = self.get_memory_percent()
        available_gb = self.get_available_memory_gb()
        
        if mem_percent >= self.CRITICAL_THRESHOLD:
            status = 'critical'
        elif mem_percent >= self.HIGH_THRESHOLD:
            status = 'high'
        elif mem_percent < self.LOW_THRESHOLD:
            status = 'low'
        else:
            status = 'normal'
        
        return {
            'percent_used': mem_percent,
            'available_gb': available_gb,
            'status': status,
            'can_load_model': mem_percent < self.HIGH_THRESHOLD
        }


class ModelPreloader:
    """
    Handles intelligent model preloading during idle times.
    """
    
    def __init__(self, inference_engine=None):
        self.inference_engine = inference_engine
        self.memory_monitor = MemoryMonitor.get_instance()
        self._preload_scheduled = False
        self._preferred_model_path: Optional[str] = None
    
    def set_preferred_model(self, model_path: str):
        """Set the model that should be preloaded when memory is available."""
        self._preferred_model_path = model_path
    
    def schedule_preload(self, model_path: str):
        """
        Schedule a model to be preloaded when memory allows.
        """
        self._preferred_model_path = model_path
        self._preload_scheduled = True
        
        # Set up callback for when memory becomes available
        self.memory_monitor.set_callbacks(
            on_critical=self._on_memory_critical,
            on_available=self._on_memory_available
        )
        
        # Start monitoring if not already
        self.memory_monitor.start_monitoring()
        logger.info(f"Scheduled preload for: {model_path}")
    
    def _on_memory_critical(self):
        """Handle critical memory - unload model."""
        if self.inference_engine and self.inference_engine.is_loaded():
            logger.warning("Memory critical - unloading model")
            self.inference_engine.unload_model()
    
    def _on_memory_available(self):
        """Handle available memory - try to preload."""
        if not self._preload_scheduled or not self._preferred_model_path:
            return
        
        if self.inference_engine and not self.inference_engine.is_loaded():
            logger.info("Memory available - preloading model")
            try:
                self.inference_engine.load_model(self._preferred_model_path)
                self._preload_scheduled = False
            except Exception as e:
                logger.error(f"Preload failed: {e}")
