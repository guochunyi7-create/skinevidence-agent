"""访客隔离与历史持久化单测。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.services.history_store import HistoryStore, normalize_vid


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = HistoryStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_visitor_isolation(self):
        self.store.append("v-alice123456", "A的问题", "A的答复", "other")
        self.store.append("v-bob99999999", "B的问题", "B的答复", "other")
        alice = self.store.sessions("v-alice123456")
        bob = self.store.sessions("v-bob99999999")
        self.assertEqual(len(alice), 1)
        self.assertEqual(len(bob), 1)
        self.assertEqual(alice[0]["title"], "A的问题")
        self.assertEqual(bob[0]["title"], "B的问题")

    def test_sessions_grouping_and_detail(self):
        self.store.append("v-group123456", "问题1", "答1", "other", session_id="s1")
        self.store.append("v-group123456", "问题2", "答2", "other", session_id="s1")
        self.store.append("v-group123456", "问题3", "答3", "other", session_id="s2")
        sessions = self.store.sessions("v-group123456")
        self.assertEqual(len(sessions), 2)
        s1 = next(s for s in sessions if s["session_id"] == "s1")
        self.assertEqual(s1["turns"], 2)
        detail = self.store.session_detail("v-group123456", "s1")
        self.assertEqual([d["question"] for d in detail], ["问题1", "问题2"])

    def test_formula_flag_and_stats(self):
        self.store.append("v-formula123", "研制配方", "草案", "formulation",
                          {"formula": {"product_label": "眼霜"}, "needs_human": False})
        s = self.store.sessions("v-formula123")[0]
        self.assertTrue(s["has_formula"])
        self.assertEqual(self.store.stats("v-formula123")["formulas"], 1)

    def test_delete_session(self):
        self.store.append("v-del1234567", "问题1", "答1", "other", session_id="s1")
        self.store.append("v-del1234567", "问题2", "答2", "other", session_id="s2")
        self.assertTrue(self.store.delete_session("v-del1234567", "s1"))
        self.assertEqual(len(self.store.sessions("v-del1234567")), 1)
        self.assertFalse(self.store.delete_session("v-del1234567", "not-exist"))

    def test_vid_normalization(self):
        self.assertEqual(normalize_vid(""), "legacy")
        self.assertEqual(normalize_vid("  v-abc123  "), "v-abc123")
        with self.assertRaises(ValueError):
            normalize_vid("bad!!!id")

    def test_corrupt_line_is_skipped(self):
        path = self.store._file("v-corrupt123")
        with path.open("a", encoding="utf-8") as fh:
            fh.write("{坏掉的json}\n")
        self.store.append("v-corrupt123", "正常问题", "正常答复", "other")
        sessions = self.store.sessions("v-corrupt123")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["title"], "正常问题")


if __name__ == "__main__":
    unittest.main()
