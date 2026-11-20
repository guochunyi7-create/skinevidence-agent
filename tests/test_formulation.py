"""配方研制能力单测：引擎计算、合规校验、路由区分。

不调用任何外部模型，零 token 消耗。
"""

from __future__ import annotations

import unittest

from app.core.config import Settings
from app.services.formulation_service import FormulationService
from app.services.router import QuestionRouter


class FormulationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = FormulationService()

    def test_library_loaded(self):
        self.assertGreaterEqual(len(self.fs.ingredients), 40)
        self.assertGreaterEqual(len(self.fs.templates), 5)
        self.assertIsNotNone(self.fs.get("玻色因"))
        self.assertIsNotNone(self.fs.get("Niacinamide"))
        self.assertIsNotNone(self.fs.get("439685-79-7"))

    def test_search_by_benefit(self):
        names = [i.zh for i in self.fs.search(benefit="抗皱", limit=8)]
        self.assertIn("玻色因", names)
        self.assertIn("乙酰基六肽-8", names)

    def test_build_closes_to_100_and_fills_required(self):
        draft = self.fs.build(
            [{"name": "玻色因", "pct": 3}, {"name": "透明质酸钠"}],
            product_type="eye_mask",
            brief="抗皱眼膜",
        )
        self.assertEqual(draft.total, 100.0)
        cats = {row.category for row in draft.items}
        self.assertIn("防腐剂", cats)
        self.assertIn("螯合剂", cats)
        self.assertIn("成膜剂", cats)
        self.assertEqual(draft.items[0].name, "水")

    def test_out_of_range_is_clamped_with_warning(self):
        draft = self.fs.build(
            [{"name": "玻色因", "pct": 99}], product_type="serum"
        )
        self.assertTrue(any("超出常规区间" in n for n in draft.risk_notes))
        row = next(r for r in draft.items if r.name == "玻色因")
        self.assertLessEqual(row.pct, 10)

    def test_conflict_and_special_cosmetic_warnings(self):
        draft = self.fs.build(
            [{"name": "卡波姆"}, {"name": "氯化钾"}, {"name": "烟酰胺"}],
            product_type="serum",
        )
        joined = " ".join(draft.risk_notes)
        self.assertIn("配伍", joined)
        self.assertIn("特殊化妆品", joined)

    def test_carbomer_auto_adds_neutralizer(self):
        draft = self.fs.build([{"name": "卡波姆"}], product_type="serum")
        self.assertTrue(any(r.name == "三乙醇胺" for r in draft.items))
        self.assertTrue(any("中和" in r for r in draft.rationale))

    def test_total_never_exceeds_100(self):
        # 用户给出明显过量的浓度时，引擎应先裁剪到常规上限，再由水补足到 100%
        draft = self.fs.build(
            [{"name": "甘油", "pct": 80}, {"name": "角鲨烷", "pct": 50}],
            product_type="cream",
        )
        self.assertEqual(draft.total, 100.0)
        gly = next(r for r in draft.items if r.name == "甘油")
        squalane = next(r for r in draft.items if r.name == "角鲨烷")
        self.assertEqual(gly.pct, 15)
        self.assertEqual(squalane.pct, 20)
        self.assertTrue(any("超出常规区间" in n for n in draft.risk_notes))


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = QuestionRouter(Settings())

    def test_development_intent_passes_to_agent(self):
        for q in [
            "帮我研制一个抗皱保湿眼霜配方",
            "开发一款舒缓精华液配方",
            "推荐一个美白配方",
            "设计一个清爽洁面配方",
            "这个配方怎么配",
        ]:
            with self.subTest(q=q):
                d = self.router.route(q)
                self.assertEqual(d.kind, "agent")
                self.assertTrue(d.formulation)

    def test_existing_formula_lookup_still_blocked(self):
        for q in ["这个眼膜的成分表是什么", "这款产品的配方是什么", "它加了什么成分"]:
            with self.subTest(q=q):
                d = self.router.route(q)
                self.assertEqual(d.kind, "formula")
                self.assertIn("研制", d.answer)

    def test_ingredient_property_not_blocked(self):
        d = self.router.route("玻色因的CAS号是多少")
        self.assertEqual(d.kind, "agent")
        self.assertFalse(d.formulation)


if __name__ == "__main__":
    unittest.main()
