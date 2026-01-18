"""
Hardware Monitor Service - Device Capabilities and Battery Management

Provides:
- Hardware detection (RAM, device type)
- Battery status monitoring
- Battery-aware request queue for low-battery throttling
"""

import logging
import platform
import threading
import time
from typing import Callable, List, Dict, Any, Optional
from collections import deque

# Try importing psutil for desktop/dev environment
try:
    import psutil
except ImportError:
    psutil = None

# Try importing plyer for cross-platform wrappers
try:
    from plyer import battery
except ImportError:
    battery = None

logger = logging.getLogger(__name__)


class HardwareMonitor:
    """
    Detects hardware capabilities to guide model selection and optimization.
    """
    
    @staticmethod
    def get_total_ram_gb() -> float:
        """Returns total RAM in GB."""
        if psutil:
            mem = psutil.virtual_memory()
            return mem.total / (1024 ** 3)
        return 4.0  # Default fallback

    @staticmethod
    def is_low_end_device() -> bool:
        """
        Heuristic: If RAM < 6GB, treat as constraints-heavy (use 4-bit quant).
        """
        ram = HardwareMonitor.get_total_ram_gb()
        logger.info(f"Detected RAM: {ram:.2f} GB")
        return ram < 6.0

    @staticmethod
    def get_battery_status() -> Optional[Dict[str, Any]]:
        """Returns battery info from plyer.battery."""
        if battery is None:
            return None
        try:
            status = battery.status
            return status
        except Exception as e:
            logger.warning(f"Battery info unavailable: {e}")
            return None
    
    @staticmethod
    def is_charging() -> bool:
        """Check if device is currently charging."""
        status = HardwareMonitor.get_battery_status()
        if status:
            return status.get('isCharging', False)
        return True  # Assume charging if unknown (don't throttle)
    
    @staticmethod
    def get_battery_percent() -> float:
        """Get battery percentage."""
        status = HardwareMonitor.get_battery_status()
        if status:
            return status.get('percentage', 100.0)
        return 100.0  # Assume full if unknown


class BatteryOptimizer:
    """
    Provides decisions based on battery state.
    """
    
    LOW_BATTERY_THRESHOLD = 20  # Throttle below this %
    CRITICAL_BATTERY_THRESHOLD = 10  # Queue requests below this %
    
    @classmethod
    def should_throttle(cls) -> bool:
        """Returns True if battery is low (< 20%) and not charging."""
        status = HardwareMonitor.get_battery_status()
        if not status:
            return False
        
        is_charging = status.get('isCharging', False)
        percent = status.get('percentage', 100)
        
        if not is_charging and percent < cls.LOW_BATTERY_THRESHOLD:
            logger.warning("Low battery detected. Throttling recommended.")
            return True
        return False
    
    @classmethod
    def should_queue_requests(cls) -> bool:
        """Returns True if battery is critical - queue instead of immediate process."""
        status = HardwareMonitor.get_battery_status()
        if not status:
            return False
        
        is_charging = status.get('isCharging', False)
        percent = status.get('percentage', 100)
        
        return not is_charging and percent < cls.CRITICAL_BATTERY_THRESHOLD
    
    @classmethod
    def get_power_mode(cls) -> str:
        """
        Get current power mode based on battery state.
        
        Returns:
            'full': Full performance (charging or high battery)
            'balanced': Reduced performance (20-50% battery)
            'powersave': Aggressive power saving (< 20% battery)
        """
        status = HardwareMonitor.get_battery_status()
        if not status:
            return 'full'
        
        is_charging = status.get('isCharging', False)
        percent = status.get('percentage', 100)
        
        if is_charging or percent >= 50:
            return 'full'
        elif percent >= cls.LOW_BATTERY_THRESHOLD:
            return 'balanced'
        else:
            return 'powersave'


class BatteryAwareQueue:
    """
    Queues inference requests during low battery and processes when charging.
    Implements batch processing to reduce CPU wake cycles.
    """
    
    def __init__(self, process_callback: Optional[Callable] = None):
        self._queue: deque = deque()
        self._process_callback = process_callback
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Batch processing settings
        self.batch_size = 5
        self.check_interval = 10.0  # seconds
    
    def set_process_callback(self, callback: Callable[[List[Dict]], None]):
        """Set callback for processing queued items."""
        self._process_callback = callback
    
    def enqueue(self, request: Dict[str, Any]) -> bool:
        """
        Add request to queue if in power-save mode.
        
        Returns:
            True if queued, False if should process immediately
        """
        power_mode = BatteryOptimizer.get_power_mode()
        
        if power_mode == 'powersave':
            with self._lock:
                self._queue.append({
                    'request': request,
                    'timestamp': time.time()
                })
            logger.info(f"Request queued (power-save mode). Queue size: {len(self._queue)}")
            return True
        
        return False
    
    def get_queue_size(self) -> int:
        """Get number of pending requests."""
        return len(self._queue)
    
    def start_monitoring(self):
        """Start monitoring for charging state to process queue."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="BatteryQueueMonitor"
        )
        self._monitor_thread.start()
        logger.info("Battery queue monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
    
    def _monitor_loop(self):
        """Monitor for charging state and process queue."""
        while self._monitoring:
            try:
                if HardwareMonitor.is_charging() and self._queue:
                    self._process_batch()
            except Exception as e:
                logger.error(f"Battery queue error: {e}")
            
            time.sleep(self.check_interval)
    
    def _process_batch(self):
        """Process a batch of queued requests."""
        if not self._process_callback:
            return
        
        batch = []
        with self._lock:
            for _ in range(min(self.batch_size, len(self._queue))):
                if self._queue:
                    item = self._queue.popleft()
                    batch.append(item['request'])
        
        if batch:
            logger.info(f"Processing batch of {len(batch)} queued requests")
            try:
                self._process_callback(batch)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Re-queue failed items
                with self._lock:
                    for req in batch:
                        self._queue.appendleft({'request': req, 'timestamp': time.time()})
    
    def force_process_all(self):
        """Force process all queued items immediately."""
        if not self._process_callback:
            return
        
        with self._lock:
            all_requests = [item['request'] for item in self._queue]
            self._queue.clear()
        
        if all_requests:
            logger.info(f"Force processing {len(all_requests)} queued requests")
            self._process_callback(all_requests)
