"""零密钥配方研制 Demo（面试官快速体验用）。

不需要任何 API Key、不需要启动服务、不需要原始 PDF，直接调用配方引擎，
演示「选料检索 → 剂量计算 → 骨架补齐 → 闭合 100% → 合规校验」全过程。

    python scripts/offline_demo.py

它会把每一步的输出打印出来，并模拟 Agent 的工具调用轨迹，
让观看者直观看到：模型负责选料决策，引擎负责所有剂量计算。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.formulation_service import FormulationService

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "data/formulation/ingredient_library.yaml"
TPL = ROOT / "data/formulation/product_templates.yaml"

LINE = "=" * 72


def show_trajectory(steps: list[tuple[str, dict, str]]) -> None:
    print("\n【Agent 工具调用轨迹】")
    for i, (tool, args, summary) in enumerate(steps, 1):
        print(f"  {i}. {tool}({json.dumps(args, ensure_ascii=False)})")
        print(f"     → {summary}")


def demo_formulation(fs: FormulationService, brief: str, benefit: str, product_type: str, picks: list[str]) -> None:
    print(LINE)
    print(f"需求：{brief}")
    print(LINE)

    steps: list[tuple[str, dict, str]] = []

    # 第 1 步：按功效检索候选原料（模拟 search_formulation_ingredients）
    candidates = fs.search(benefit=benefit, limit=6)
    cand_str = "、".join(f"{x.zh}({x.low:g}~{x.high:g}%)" for x in candidates[:5])
    steps.append((
        "search_formulation_ingredients",
        {"benefit": benefit},
        f"命中 {len(candidates)} 种候选：{cand_str}",
    ))

    # 第 2 步：确认选中原料详情（模拟 get_formulation_ingredient）
    for name in picks:
        ing = fs.get(name)
        if ing:
            steps.append((
                "get_formulation_ingredient",
                {"name": name},
                f"{ing.zh} | 常规 {ing.low:g}~{ing.high:g}% | 常用 {ing.typical:g}% | 功效 {'/'.join(ing.benefits)}",
            ))

    # 第 3 步：确定性构建配方草案（build_formula，纯代码计算）
    draft = fs.build(
        items=[{"name": n} for n in picks],
        product_type=product_type,
        brief=brief,
    )
    steps.append((
        "build_formula",
        {"product_type": product_type, "items": picks},
        f"生成 {draft.product_label}草案，共 {len(draft.items)} 项，合计 {draft.total}%（已闭合至 100%）",
    ))

    show_trajectory(steps)

    # 结构化成分表
    print(f"\n【配方草案 · {draft.product_label}】")
    print(f"  {'分类':<8}{'原料':<16}{'用量':>8}   作用")
    print(f"  {'-'*8}{'-'*16}{'-'*8}   {'-'*24}")
    for it in draft.items:
        print(f"  {it.category:<8}{it.name:<16}{str(it.pct)+'%':>8}   {it.role}")

    if draft.rationale:
        print("\n【系统自动补齐（确定性规则）】")
        for r in draft.rationale:
            print(f"  · {r}")

    if draft.risk_notes:
        print("\n【合规 / 配伍风险提示】")
        for r in draft.risk_notes:
            print(f"  ! {r}")

    print(f"\n  合计：{draft.total}%  ✓ 已闭合至 100%")
    print(f"\n  免责声明：{draft.disclaimer}")


def main() -> None:
    fs = FormulationService(str(LIB), str(TPL))
    print(LINE)
    print("  SkinEvidence · 配方研制引擎 离线演示（零密钥）")
    print(f"  已加载 {len(fs.ingredients)} 种原料 / {len(fs.templates)} 类剂型")
    print(LINE)

    demo_formulation(
        fs,
        brief="帮我做一款抗皱保湿眼霜配方",
        benefit="抗皱",
        product_type="eye_cream",
        picks=["玻色因", "视黄醇", "透明质酸钠"],
    )
    demo_formulation(
        fs,
        brief="做一款舒缓提亮的精华液",
        benefit="舒缓",
        product_type="serum",
        picks=["积雪草提取物", "烟酰胺", "泛醇"],
    )

    print("\n" + LINE)
    print("  以上所有百分比均由确定性代码计算，模型无法编造浓度。")
    print("  完整 Agent 模式（模型自主选料 + 工具编排）见 README「方式二」。")
    print(LINE)


if __name__ == "__main__":
    main()
