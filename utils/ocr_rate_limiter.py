import time
import threading
from typing import Tuple, Dict
import psutil
import logging

logger = logging.getLogger(__name__)

class OCRRateLimiter:
    """
    Token bucket rate limiter with aggressive memory management.
    """
    
    def __init__(
        self,
        max_requests_per_minute: int = 3,
        max_batch_files: int = 5,
        max_file_size_mb: int = 20,
        memory_threshold_mb: int = 2048,  # RAISED to 2GB
        memory_critical_mb: int = 900
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_batch_files = max_batch_files
        self.max_file_size_mb = max_file_size_mb
        self.memory_threshold_mb = memory_threshold_mb
        self.memory_critical_mb = memory_critical_mb  # now configurable
        
        self.request_times: Dict[str, list] = {}
        self.active_requests: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def check_rate_limit(self, client_ip: str) -> Tuple[bool, str, int]:
        """
        Check if request should be allowed.
        Returns: (allowed: bool, reason: str, retry_after_seconds: int)
        """
        now = time.time()
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        
        # If memory is critically high, try to cleanup and retry
        if memory_mb > self.memory_critical_mb:
            print(f"CRITICAL MEMORY ({memory_mb:.1f}MB > {self.memory_critical_mb}MB). Forcing cleanup...")
            try:
                from services.ocr_service import _aggressive_model_unload
                _aggressive_model_unload()
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"After cleanup: {memory_mb:.1f}MB")
            except Exception as e:
                print(f"Cleanup failed: {e}")
        
        with self.lock:
            # Cleanup old requests
            cutoff = now - 60
            if client_ip in self.request_times:
                self.request_times[client_ip] = [
                    t for t in self.request_times[client_ip] if t > cutoff
                ]
            
            current_count = len(self.request_times.get(client_ip, []))
            
            # Check rate limit
            if current_count >= self.max_requests_per_minute:
                oldest_time = self.request_times[client_ip][0]
                retry_after = int(60 - (now - oldest_time)) + 1
                return (
                    False,
                    f"OCR rate limit exceeded ({self.max_requests_per_minute} per minute)",
                    retry_after
                )
            
            # Check memory (re-check after potential cleanup)
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if memory_mb > self.memory_threshold_mb:
                return (
                    False,
                    f"Server memory too high ({memory_mb:.0f}MB / {self.memory_threshold_mb}MB). Waiting for cleanup.",
                    30
                )
            
            # Check active concurrent requests
            active = self.active_requests.get(client_ip, 0)
            if active > 1:  # Max 1 concurrent per IP
                return (
                    False,
                    "Only 1 concurrent OCR request allowed per IP",
                    5
                )
            
            # Record this request
            if client_ip not in self.request_times:
                self.request_times[client_ip] = []
            self.request_times[client_ip].append(now)
            self.active_requests[client_ip] = active + 1
            
            return True, "OK", 0
    
    def check_batch_size(self, file_count: int) -> Tuple[bool, str]:
        """Validate batch request size"""
        if file_count > self.max_batch_files:
            return (
                False,
                f"Batch size too large (max {self.max_batch_files} files, got {file_count})"
            )
        return True, "OK"
    
    def check_file_size(self, file_size_bytes: int) -> Tuple[bool, str]:
        """Validate individual file size"""
        file_size_mb = file_size_bytes / 1024 / 1024
        if file_size_mb > self.max_file_size_mb:
            return (
                False,
                f"File too large ({file_size_mb:.1f}MB, max {self.max_file_size_mb}MB)"
            )
        return True, "OK"
    
    def release_request(self, client_ip: str):
        """Call when OCR request completes"""
        with self.lock:
            self.active_requests[client_ip] = max(0, self.active_requests.get(client_ip, 1) - 1)

# Singleton with new thresholds
ocr_limiter = OCRRateLimiter(
    max_requests_per_minute=3,
    max_batch_files=5,
    max_file_size_mb=20,
    memory_threshold_mb=3400,   # recommend 3.4GB
    memory_critical_mb=3000     # recommend 3.0GB (trigger cleanup)
)