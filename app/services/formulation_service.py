"""配方研制引擎：把「选料意图」变成可计算的配方草案。

职责边界（与 LLM 分工）：
  - LLM 负责决策：理解 brief、决定选哪些料、组织语言解释；
  - 本引擎负责计算：剂量取值、区间裁剪、必需组分补齐、基底补足至 100%、
    配伍冲突与合规提示，并产出结构化 FormulaDraft。

所有用量来自 data/formulation/ingredient_library.yaml 的行业常规参考区间，
均由本引擎按确定性规则计算，模型不能凭空捏造浓度。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.schemas.chat import FormulaDraft, FormulaItem

logger = logging.getLogger(__name__)

_CATEGORY_ORDER = [
    "base", "humectant", "thickener", "film", "active", "soothing",
    "emollient", "emulsifier", "preservative", "chelator", "ph", "surfactant",
]

# 模型常用的口语/旧称 → 库内标准名。避免把同一个原料误判为"未收录"。
_EXTRA_ALIASES = {
    "透明质酸": "透明质酸钠",
    "玻尿酸": "透明质酸钠",
    "维生素e": "生育酚",
    "维e": "生育酚",
    "维生素c酯": "3-O-乙基抗坏血酸",
    "维c衍生物": "3-O-乙基抗坏血酸",
    "丙三醇": "甘油",
    "1,3-丁二醇": "丁二醇",
    "角鲨烯": "角鲨烷",
}


@dataclass(frozen=True)
class Ingredient:
    zh: str
    en: str
    cas: str
    cat: str
    benefits: tuple[str, ...]
    low: float
    high: float
    typical: float
    phase: str
    process: str
    compat: str
    caveat: str
    evidence: str
    special: bool = False
    avoid: tuple[str, ...] = ()

    @property
    def role(self) -> str:
        return "、".join(self.benefits)

    def describe(self, category_label: str = "") -> str:
        return (
            f"{self.zh}（{self.en}，CAS {self.cas or '—'}）| {category_label or self.cat} | "
            f"功效：{self.role} | 常规 {self.low}~{self.high}%，常用 {self.typical}% | "
            f"相态：{self.phase} | 工艺：{self.process} | 配伍：{self.compat} | "
            f"注意：{self.caveat} | 依据：{'MSDS 已收录' if self.evidence == 'msds' else '行业常规参考'}"
        )


@dataclass
class _Template:
    label: str
    base: str
    required: tuple[str, ...]
    recommended: tuple[str, ...]
    oil_range: tuple[float, float] | None
    note: str


@dataclass
class _Picked:
    ing: Ingredient
    pct: float


class FormulationService:
    def __init__(
        self,
        library_file: str | Path = "data/formulation/ingredient_library.yaml",
        template_file: str | Path = "data/formulation/product_templates.yaml",
    ) -> None:
        lib = self._load(library_file)
        tpl = self._load(template_file)

        self.disclaimer: str = lib.get("disclaimer", "")
        self.category_labels: dict[str, str] = lib.get("categories", {}) or {}

        self.ingredients: dict[str, Ingredient] = {}
        self._alias: dict[str, str] = {}
        for row in lib.get("ingredients", []) or []:
            low, high, typical = (list(row.get("use") or [0, 0, 0]) + [0, 0, 0])[:3]
            ing = Ingredient(
                zh=str(row.get("zh") or "").strip(),
                en=str(row.get("en") or "").strip(),
                cas=str(row.get("cas") or "").strip(),
                cat=str(row.get("cat") or "other").strip(),
                benefits=tuple(row.get("benefits") or ()),
                low=float(low), high=float(high), typical=float(typical),
                phase=str(row.get("phase") or ""),
                process=str(row.get("process") or ""),
                compat=str(row.get("compat") or ""),
                caveat=str(row.get("caveat") or ""),
                evidence=str(row.get("evidence") or "industry"),
                special=bool(row.get("special", False)),
                avoid=tuple(row.get("avoid") or ()),
            )
            if not ing.zh:
                continue
            self.ingredients[ing.zh] = ing
            for key in (ing.zh, ing.en, ing.cas):
                if key and key != "0":
                    self._alias[key.strip().lower()] = ing.zh
        for alias, target in _EXTRA_ALIASES.items():
            if target in self.ingredients:
                self._alias[alias] = target

        self.defaults: dict[str, str] = tpl.get("defaults", {}) or {}
        self.special_pairs: dict[str, dict] = tpl.get("special_pairs", {}) or {}
        self.templates: dict[str, _Template] = {}
        for key, row in (tpl.get("templates") or {}).items():
            oil = row.get("oil_range")
            self.templates[key] = _Template(
                label=str(row.get("label") or key),
                base=str(row.get("base") or "水"),
                required=tuple(row.get("required") or ()),
                recommended=tuple(row.get("recommended") or ()),
                oil_range=(float(oil[0]), float(oil[1])) if oil else None,
                note=str(row.get("note") or ""),
            )
        logger.info("配方库加载完成：%d 种原料，%d 个剂型模板", len(self.ingredients), len(self.templates))

    @staticmethod
    def _load(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    # ---- 查询 ----

    def get(self, name: str) -> Ingredient | None:
        """按中文名 / 英文名 / CAS 定位原料。"""
        key = str(name or "").strip().lower()
        zh = self._alias.get(key)
        return self.ingredients.get(zh) if zh else None

    def search(
        self,
        query: str = "",
        category: str | None = None,
        benefit: str | None = None,
        limit: int = 8,
    ) -> list[Ingredient]:
        """按功能分类 / 功效标签 / 自由文本检索候选原料。"""
        cat_key = category
        if cat_key and cat_key not in self.ingredients and cat_key not in self.category_labels:
            for key, label in self.category_labels.items():
                if label == cat_key:
                    cat_key = key
                    break

        tokens = [t for t in re.split(r"[\s,，、]+", query.strip().lower()) if t] if query else []
        found: list[Ingredient] = []
        for ing in self.ingredients.values():
            if cat_key and ing.cat != cat_key:
                continue
            if benefit and benefit not in ing.benefits and benefit not in ing.role:
                continue
            if tokens:
                hay = " ".join([
                    ing.zh, ing.en, ing.cas, self.category_labels.get(ing.cat, ing.cat),
                    ing.role, ing.process, ing.phase,
                ]).lower()
                if not any(tok in hay for tok in tokens):
                    continue
            found.append(ing)
        return found[: max(1, limit)]

    def format_candidates(self, ingredients: list[Ingredient]) -> str:
        if not ingredients:
            return "原料库中未找到匹配原料。可换一种功效说法，或说明剂型后让我从常用原料中推荐。"
        lines = [f"候选原料 {len(ingredients)} 种（用量为行业常规参考区间，非法规限值）："]
        for ing in ingredients:
            lines.append("- " + ing.describe(self.category_labels.get(ing.cat, ing.cat)))
        return "\n".join(lines)

    # ---- 构建 ----

    def build(
        self,
        items: list[dict[str, Any]] | None,
        product_type: str = "serum",
        brief: str = "",
    ) -> FormulaDraft:
        """根据选料意图与剂型模板，确定性生成配方草案。"""
        tpl = self.templates.get(product_type) or self.templates.get("serum")
        if tpl is None:
            raise RuntimeError("配方引擎未加载任何剂型模板")

        warnings: list[str] = []
        rationale: list[str] = []
        picked: dict[str, _Picked] = {}

        # 1) 解析模型选料
        for raw in items or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if not name:
                continue
            ing = self.get(name)
            if ing is None:
                warnings.append(f"原料库未收录「{name}」，已跳过。请确认名称或补充该原料资料。")
                continue
            pct_raw = raw.get("pct")
            try:
                pct = float(pct_raw) if pct_raw not in (None, "") else ing.typical
            except (TypeError, ValueError):
                pct = ing.typical
                warnings.append(f"「{name}」用量格式无法识别，已按常用值 {ing.typical}% 处理。")
            if pct < ing.low or pct > ing.high:
                fixed = min(max(pct, ing.low), ing.high)
                warnings.append(
                    f"「{name}」指定 {pct:g}% 超出常规区间 {ing.low:g}~{ing.high:g}%，"
                    f"已按 {fixed:g}% 计入，实际用量须研发确认。"
                )
                pct = fixed
            picked[ing.zh] = _Picked(ing=ing, pct=round(pct, 4))

        # 2) 按剂型骨架补齐必需组分
        for cat in tpl.required:
            if any(p.ing.cat == cat for p in picked.values()):
                continue
            default_name = self.defaults.get(cat)
            ing = self.get(default_name) if default_name else None
            if ing is None:
                for candidate in self.ingredients.values():
                    if candidate.cat == cat:
                        ing = candidate
                        break
            if ing:
                picked[ing.zh] = _Picked(ing=ing, pct=ing.typical)
                rationale.append(
                    f"按{tpl.label}骨架补入{self.category_labels.get(cat, cat)}「{ing.zh}」"
                    f"{ing.typical:g}%（系统补齐）"
                )

        # 3) 特殊配伍对（如卡波姆需中和）
        for trigger, pair in self.special_pairs.items():
            if trigger not in picked:
                continue
            add_name = str(pair.get("add") or "")
            if add_name and add_name not in picked:
                add_ing = self.get(add_name)
                if add_ing:
                    picked[add_ing.zh] = _Picked(ing=add_ing, pct=add_ing.typical)
                    rationale.append(f"{pair.get('note', '')}；补入「{add_ing.zh}」{add_ing.typical:g}%")

        # 4) 配伍冲突
        names = set(picked)
        for name, p in picked.items():
            for other in p.ing.avoid:
                if other in names:
                    warnings.append(f"配伍提示：「{name}」与「{other}」存在已知配伍风险，须研发验证。")

        # 5) 特殊化妆品合规提示
        for name, p in picked.items():
            if p.ing.special:
                warnings.append(
                    f"合规提示：「{name}」若用于美白/祛斑等宣称，可能按特殊化妆品管理，须法规确认。"
                )

        # 6) 总量闭合：基底补足至 100%，超限则按比例压缩
        used = sum(p.pct for p in picked.values())
        if used > 100:
            scale = 95.0 / used
            for p in picked.values():
                p.pct = round(p.pct * scale, 4)
            warnings.append(f"原料合计 {used:.2f}% 超过 100%，已按比例压缩至 95%，请重新核对。")
            used = 95.0
        water = self.get(tpl.base)
        water_pct = round(100.0 - used, 4)
        if water and water_pct > 0:
            picked[water.zh] = _Picked(ing=water, pct=water_pct)

        # 7) 油相区间提示
        if tpl.oil_range:
            oil = sum(p.pct for p in picked.values() if p.ing.cat == "emollient")
            lo, hi = tpl.oil_range
            if oil < lo or oil > hi:
                warnings.append(
                    f"肤感提示：油相合计 {oil:.1f}%，不在{tpl.label}建议区间 {lo:g}~{hi:g}% 内，"
                    "可能与目标肤感有偏差。"
                )

        # 8) 输出结构化草案（按配方习惯排序）
        ordered = sorted(
            picked.values(),
            key=lambda p: (
                _CATEGORY_ORDER.index(p.ing.cat) if p.ing.cat in _CATEGORY_ORDER else 99,
                -p.pct,
            ),
        )
        rows: list[FormulaItem] = []
        for p in ordered:
            note = p.ing.caveat
            if p.ing.evidence == "msds":
                note = (note + "；知识库有该原料 MSDS 可查").strip("；")
            rows.append(
                FormulaItem(
                    name=p.ing.zh,
                    category=self.category_labels.get(p.ing.cat, p.ing.cat),
                    pct=round(p.pct, 3),
                    role=p.ing.role,
                    note=note,
                )
            )

        return FormulaDraft(
            product_type=product_type,
            product_label=tpl.label,
            brief=brief,
            items=rows,
            total=round(sum(r.pct for r in rows), 2),
            rationale=rationale,
            risk_notes=warnings,
            disclaimer=self.disclaimer,
        )
