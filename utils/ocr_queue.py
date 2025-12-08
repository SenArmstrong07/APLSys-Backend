
import asyncio
from typing import Optional

class OCRQueue:
    """Queue OCR jobs to avoid memory spikes"""
    def __init__(self, max_concurrent: int = 1):
        self.queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.active = 0
    
    async def submit(self, job_func, *args):
        """Submit job to queue"""
        await self.queue.put((job_func, args))
        return await self._process()
    
    async def _process(self):
        """Process queue with concurrency limit"""
        if self.active >= self.max_concurrent:
            await asyncio.sleep(0.1)
            return await self._process()
        
        self.active += 1
        try:
            job_func, args = await asyncio.wait_for(
                self.queue.get(), timeout=300
            )
            return await job_func(*args)
        finally:
            self.active -= 1

ocr_queue = OCRQueue(max_concurrent=1)