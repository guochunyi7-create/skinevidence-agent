"""问题类型路由单测：配方类问题统一答复，原料属性问题不受误伤。"""

from __future__ import annotations

import unittest

from app.core.config import Settings
from app.services.router import FORMULA_ANSWER, QuestionRouter


def _router() -> QuestionRouter:
    return QuestionRouter(Settings(
        compliance_file="data/compliance/banned_claims.yaml",
        glossary_file="data/knowledge/glossary/ingredients.yaml",
        index_file="data/index/chunks.jsonl",
    ))


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.r = _router()

    def test_formula_questions(self):
        for q in [
            "这个眼膜的配方是什么",
            "玻色因眼膜的成分表",
            "产品里含有哪些成分",
            "这个精华的配料表",
            "面霜的成分组成",
        ]:
            with self.subTest(q=q):
                d = self.r.route(q)
                self.assertEqual(d.kind, "formula", q)
                self.assertEqual(d.answer, FORMULA_ANSWER)

    def test_ingredient_property_not_blocked(self):
        """问单个原料属性，即便带"成分"二字也应走 Agent，不被误伤。"""
        for q in [
            "玻色因成分安全吗",
            "透明质酸钠的CAS号是多少",
            "甘油的毒理数据",
            "铅的限量是多少",
        ]:
            with self.subTest(q=q):
                self.assertEqual(self.r.route(q).kind, "agent", q)

    def test_normal_questions(self):
        for q in ["订单 202611080001 到哪了", "电子皮肤做过哪些认证"]:
            with self.subTest(q=q):
                self.assertEqual(self.r.route(q).kind, "agent", q)


if __name__ == "__main__":
    unittest.main(verbosity=2)
