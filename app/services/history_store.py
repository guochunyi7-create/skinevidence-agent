"""按访客隔离的对话历史持久化。

设计要点（参考 skinevidence 的访客隔离经验）：
  - 每个访客一个目录 artifacts/visitors/<vid>/，互不可见；
  - 每次问答追加一条 JSONL 记录（问题、答复、意图、草案、轨迹、时间）；
  - 同一访客的对话以 session_id 分组，"新会话"只是换 session_id，旧记录仍在；
  - 访客 ID 由前端 localStorage 生成，服务端只做格式校验，不信任其内容；
  - 历史读写失败不影响主流程——它是观测能力，不该让问答挂掉。
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_PUBLIC_VID = "legacy"


def normalize_vid(raw: str | None) -> str:
    """校验访客 ID：空值落公共桶，非法值拒绝（交给调用方转成 400）。"""
    vid = (raw or "").strip()
    if not vid:
        return _PUBLIC_VID
    if not _VID_RE.match(vid):
        raise ValueError("visitor_id 非法")
    return vid


class HistoryStore:
    def __init__(self, artifact_dir: str | Path = "artifacts") -> None:
        self.root = Path(artifact_dir) / "visitors"

    def _file(self, vid: str) -> Path:
        safe = normalize_vid(vid)
        path = self.root / safe / "history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ---- 写 ----

    def append(
        self,
        vid: str,
        question: str,
        answer: str,
        intent: str,
        response: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> str:
        """追加一条问答记录，返回其 record_id。"""
        sid = session_id or uuid.uuid4().hex[:16]
        if not _SESSION_RE.match(sid):
            sid = uuid.uuid4().hex[:16]
        record = {
            "record_id": uuid.uuid4().hex[:16],
            "session_id": sid,
            "ts": round(time.time(), 3),
            "question": question,
            "answer": answer,
            "intent": intent,
            "needs_human": bool((response or {}).get("needs_human", False)),
            "confidence": float((response or {}).get("confidence", 0.0)),
            "formula": (response or {}).get("formula"),
            "trajectory": (response or {}).get("trajectory", []),
            "citations": (response or {}).get("citations", []),
        }
        try:
            with self._file(vid).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("历史写入失败（不影响问答）: %s", exc)
        return record["record_id"]

    # ---- 读 ----

    def _read_all(self, vid: str) -> list[dict[str, Any]]:
        path = self._file(vid)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("跳过损坏的历史行: %s", line[:80])
        return rows

    def sessions(self, vid: str, limit: int = 50) -> list[dict[str, Any]]:
        """按 session_id 聚合，最近的会话排前面。"""
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in self._read_all(vid):
            sid = row.get("session_id") or "unknown"
            groups.setdefault(sid, []).append(row)

        out: list[dict[str, Any]] = []
        for sid, rows in groups.items():
            rows.sort(key=lambda r: r.get("ts", 0))
            first, last = rows[0], rows[-1]
            out.append({
                "session_id": sid,
                "title": (first.get("question") or "")[:60],
                "updated_at": last.get("ts", 0),
                "turns": len(rows),
                "has_formula": any(r.get("formula") for r in rows),
            })
        out.sort(key=lambda s: s["updated_at"], reverse=True)
        return out[:limit]

    def session_detail(self, vid: str, session_id: str) -> list[dict[str, Any]]:
        rows = [r for r in self._read_all(vid) if r.get("session_id") == session_id]
        rows.sort(key=lambda r: r.get("ts", 0))
        return rows

    def stats(self, vid: str) -> dict[str, Any]:
        rows = self._read_all(vid)
        sids = {r.get("session_id") for r in rows}
        return {
            "sessions": len(sids),
            "turns": len(rows),
            "formulas": sum(1 for r in rows if r.get("formula")),
        }

    # ---- 删 ----

    def delete_session(self, vid: str, session_id: str) -> bool:
        path = self._file(vid)
        if not path.exists():
            return False
        rows = self._read_all(vid)
        kept = [r for r in rows if r.get("session_id") != session_id]
        if len(kept) == len(rows):
            return False
        try:
            tmp = path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                for r in kept:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("历史删除失败: %s", exc)
            return False
        return True

    def clear_all(self, vid: str) -> bool:
        path = self._file(vid)
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as exc:
            logger.warning("历史清空失败: %s", exc)
            return False
