import time
import psutil
from ocr_rate_limiter import ocr_limiter

# Save/restore psutil.Process
_orig_process = psutil.Process

def _set_mem_mb(mb: float):
    class DummyProc:
        def __init__(self, mb): self._mb = mb
        def memory_info(self):
            class MI:
                rss = 0
            m = MI()
            m.rss = int(self._mb * 1024 * 1024)
            return m
    psutil.Process = lambda pid=None: DummyProc(mb)

def _restore_mem():
    psutil.Process = _orig_process

def _reset_limiter():
    ocr_limiter.request_times.clear()
    ocr_limiter.active_requests.clear()

def run_tests():
    try:
        print("1) Low memory -> expect allowed")
        _reset_limiter()
        _set_mem_mb(100)
        ok, reason, retry = ocr_limiter.check_rate_limit("1.2.3.4")
        print("result:", ok, reason, retry)
        if ok:
            ocr_limiter.release_request("1.2.3.4")

        print("\n2) Rate exceed (3/min) -> 4th call should fail")
        _reset_limiter()
        _set_mem_mb(100)
        ip = "2.2.2.2"
        for i in range(4):
            ok, reason, retry = ocr_limiter.check_rate_limit(ip)
            print(f"call {i+1} ->", ok, reason, retry)
            # simulate request finished (release) for sequential testing
            if ok:
                ocr_limiter.release_request(ip)
            time.sleep(0.2)

        print("\n3) Memory threshold block -> expect blocked")
        _reset_limiter()
        # lower threshold to trigger easily in test
        ocr_limiter.memory_threshold_mb = 50
        _set_mem_mb(200)
        ok, reason, retry = ocr_limiter.check_rate_limit("3.3.3.3")
        print("result:", ok, reason, retry)
        if ok:
            ocr_limiter.release_request("3.3.3.3")

    finally:
        _restore_mem()

if __name__ == "__main__":
    run_tests()