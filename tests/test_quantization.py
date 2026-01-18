"""
Unit Tests for Quantization Service
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQuantizationService:
    """Tests for dynamic quantization selection."""
    
    @patch('app.services.hardware_monitor.HardwareMonitor.get_total_ram_gb')
    def test_low_ram_uses_4bit(self, mock_ram):
        """Devices with < 6GB RAM should use 4-bit quantization."""
        mock_ram.return_value = 4.0
        
        from app.services.quantization_service import QuantizationService
        
        quant = QuantizationService.get_optimal_quantization()
        assert quant == "Q4_K_M"
    
    @patch('app.services.hardware_monitor.HardwareMonitor.get_total_ram_gb')
    def test_high_ram_uses_8bit(self, mock_ram):
        """Devices with >= 6GB RAM should use 8-bit quantization."""
        mock_ram.return_value = 8.0
        
        from app.services.quantization_service import QuantizationService
        
        quant = QuantizationService.get_optimal_quantization()
        assert quant == "Q8_0"
    
    @patch('app.services.hardware_monitor.HardwareMonitor.get_total_ram_gb')
    def test_context_size_low_ram(self, mock_ram):
        """Low RAM devices should get smaller context window."""
        mock_ram.return_value = 3.0
        
        from app.services.quantization_service import QuantizationService
        
        ctx = QuantizationService.get_optimal_context_size()
        assert ctx == 1024
    
    @patch('app.services.hardware_monitor.HardwareMonitor.get_total_ram_gb')
    def test_context_size_high_ram(self, mock_ram):
        """High RAM devices should get larger context window."""
        mock_ram.return_value = 12.0
        
        from app.services.quantization_service import QuantizationService
        
        ctx = QuantizationService.get_optimal_context_size()
        assert ctx == 4096
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation for models."""
        from app.services.quantization_service import QuantizationService
        
        # 10GB full precision model with 4-bit quant should be ~3GB
        estimated = QuantizationService.estimate_memory_usage(10.0, "Q4_K_M")
        
        # 10 * 0.25 * 1.2 = 3.0
        assert 2.5 <= estimated <= 3.5
    
    @patch('app.services.hardware_monitor.HardwareMonitor.get_total_ram_gb')
    def test_can_load_model_check(self, mock_ram):
        """Test model loading feasibility check."""
        mock_ram.return_value = 8.0
        
        from app.services.quantization_service import QuantizationService
        
        # Small model should load fine
        can_load, reason = QuantizationService.can_load_model(2.0)
        assert can_load is True
        
        # Huge model shouldn't load
        can_load, reason = QuantizationService.can_load_model(10.0)
        assert can_load is False


class TestMemoryMonitor:
    """Tests for memory monitoring."""
    
    def test_singleton_pattern(self):
        """Test singleton pattern for MemoryMonitor."""
        from app.services.memory_monitor import MemoryMonitor
        
        m1 = MemoryMonitor.get_instance()
        m2 = MemoryMonitor.get_instance()
        
        assert m1 is m2
    
    @patch('psutil.virtual_memory')
    def test_memory_percent(self, mock_mem):
        """Test memory percentage retrieval."""
        mock_mem.return_value = MagicMock(percent=75.0)
        
        from app.services.memory_monitor import MemoryMonitor
        
        percent = MemoryMonitor.get_memory_percent()
        assert percent == 75.0
    
    def test_memory_status(self):
        """Test getting memory status dict."""
        from app.services.memory_monitor import MemoryMonitor
        
        status = MemoryMonitor.get_instance().get_memory_status()
        
        assert 'percent_used' in status
        assert 'available_gb' in status
        assert 'status' in status
        assert 'can_load_model' in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
