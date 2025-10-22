import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

DB_DIR = Path("data")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "tasks.db"

class TaskStore:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT,
            filename TEXT,
            status TEXT,
            progress REAL,
            details TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """)
        conn.commit()
        conn.close()

    def create_task(self, task_type: str, filename: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> int:
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tasks (task_type, filename, status, progress, details, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_type, filename, "pending", 0.0, json.dumps(details or {}), now, now)
        )
        task_id = cur.lastrowid
        if task_id is None:
            raise RuntimeError("Failed to create task ID after insertion.")
        conn.commit()
        conn.close()
        return task_id

    def update_task(self, task_id: int, status: Optional[str] = None, progress: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        now = datetime.utcnow().isoformat()
        conn = self._get_conn()
        cur = conn.cursor()
        # fetch existing details
        cur.execute("SELECT details, progress FROM tasks WHERE id = ?", (task_id,))
        row = cur.fetchone()
        if not row:
            conn.close()
            raise KeyError(f"Task {task_id} not found")
        current_details = json.loads(row[0] or "{}")
        current_progress = float(row[1] or 0.0)
        if details:
            current_details.update(details)
        new_progress = progress if progress is not None else current_progress
        if status is None:
            status = None
        cur.execute(
            "UPDATE tasks SET status = COALESCE(?, status), progress = ?, details = ?, updated_at = ? WHERE id = ?",
            (status, new_progress, json.dumps(current_details), now, task_id)
        )
        conn.commit()
        conn.close()

    def get_task(self, task_id: int) -> Dict[str, Any]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, task_type, filename, status, progress, details, created_at, updated_at FROM tasks WHERE id = ?", (task_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            raise KeyError(f"Task {task_id} not found")
        return {
            "id": row[0],
            "task_type": row[1],
            "filename": row[2],
            "status": row[3],
            "progress": row[4],
            "details": json.loads(row[5] or "{}"),
            "created_at": row[6],
            "updated_at": row[7]
        }

    def list_tasks(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        if status:
            cur.execute("SELECT id, task_type, filename, status, progress, details, created_at, updated_at FROM tasks WHERE status = ? ORDER BY id DESC LIMIT ?", (status, limit))
        else:
            cur.execute("SELECT id, task_type, filename, status, progress, details, created_at, updated_at FROM tasks ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        conn.close()
        out = []
        for row in rows:
            out.append({
                "id": row[0],
                "task_type": row[1],
                "filename": row[2],
                "status": row[3],
                "progress": row[4],
                "details": json.loads(row[5] or "{}"),
                "created_at": row[6],
                "updated_at": row[7]
            })
        return out
