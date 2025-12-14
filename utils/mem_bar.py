import threading
from typing import Dict, Any

class MemoryBar:
    def __init__(self, cap_mb: int = 10240):
        self.cap_mb = float(cap_mb)
        self.lock = threading.Lock()
        self.items: Dict[str, float] = {}  # name -> mb

    def register(self, name: str, mb: float):
        with self.lock:
            self.items[name] = float(mb)
            return self.get_usage()

    def unregister(self, name: str):
        with self.lock:
            if name in self.items:
                del self.items[name]
            return self.get_usage()

    def set_item(self, name: str, mb: float):
        with self.lock:
            self.items[name] = float(mb)
            return self.get_usage()

    def clear_all(self):
        with self.lock:
            self.items.clear()

    def get_usage(self) -> Dict[str, Any]:
        with self.lock:
            used = sum(self.items.values())
            pct = min(100.0, (used / self.cap_mb) * 100.0) if self.cap_mb > 0 else 0.0
            return {
                "cap_mb": round(self.cap_mb, 1),
                "used_mb": round(used, 1),
                "used_percent": round(pct, 2),
                "items": {k: round(v, 1) for k, v in self.items.items()}
            }

# singleton
memory_bar = MemoryBar(cap_mb=10240)