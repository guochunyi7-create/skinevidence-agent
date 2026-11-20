"""抽取式阅读器：从检索命中的资料里抽出「能直接回答问题的那一行」。

为什么需要这一层：
    检索返回的是整块资料（一份 MSDS 的某一节、法规的某一页），直接丢给用户
    就是一堆表头加半句话。而真正的答案往往只是某一行键值对（`CAS No.：9067-32-7`）
    或某一句限定语（`铅 10`）。这一层负责定位那一行、裁到窗口、按出处组织成
    人能读的短答。

它同时是真实模型不可用时的降级路径：抽取式答案不会编造，最坏是答偏，不是答假。
对法规与原料安全场景，"不编造"比"流畅"重要得多。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.glossary import Glossary

# 表头：【文件标题｜原料（英文名）｜章节】
_HEADER_END = "】"

# 查询里的功能字：按这些字切开，剩下的就是"实词块"，比 n-gram 精确得多
# 「化妆品中铅的限量是多少」→ 化妆品 / 铅 / 限量
_DELIM_CHARS = set(
    "的了吗呢吧呀啊是有在和与或及用能会要可你我他她它这那个些么什怎为如哪些几"
    "把被给对从到就都还也又很太更最再请问一下帮我想看说做"
)
_WS = re.compile(r"[\s\u3000]+")
_ASCII = re.compile(r"[A-Za-z][A-Za-z0-9\.\-]{1,}")
_NUM = re.compile(r"\d")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")
# 问"值"类问题时，带数字的行更可能是答案
_VALUE_PROBE = re.compile(r"多少|几|限值|限量|上限|最大|浓度|含量|比例|值|号|温度|熔点|沸点|密度|分子量")
# 页码、分隔线一类噪声行
_NOISE_LINE = re.compile(r"^(?:[-–—\s]*\d+[-–—\s]*|page\s*\d+.*|第\s*\d+\s*页|\*+)$", re.I)
# 联系方式、网址一类噪声行：MSDS 首页极易被当成答案
_NOISE_UNIT = re.compile(r"@|www\.|https?:|\btel\b|\bfax\b|e-?mail|p\.?o\.?\s*box", re.I)
# MSDS 每页重复的页眉页脚（含泰文页码、文件编号、修订日期）
_BOILERPLATE = re.compile(
    r"^(?:material\s+safety\s+data\s+sheets?|safety\s+data\s+sheet)$"
    r"|หน้า|\brev\.?\s*\d|page\s+\d+\s+of\s+\d+|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\bFO-[A-Z]{2}-\d+",
    re.I,
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", text.lower())


def _is_heading(text: str, section: str) -> bool:
    """判断这一行是否只是章节标题本身。

    MSDS 的 "Section 11 - Toxicological Information" 会命中属性词，得分最高，
    但拿它当答案等于什么都没说。按去掉标点后的包含关系识别。
    """
    a, b = _norm(text), _norm(section)
    if not a or not b:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 0.6 * len(long_) and short in long_
_SENT_SPLIT = re.compile(r"(?<=[。；;！？!?])\s*")


@dataclass
class DocBlock:
    title: str
    section: str
    source: str
    content: str


@dataclass
class Unit:
    """候选答案单元：一行键值对，或一句话。"""

    text: str
    doc: DocBlock
    section_hit: bool = False
    score: float = 0.0
    keys: set[str] = field(default_factory=set)
    forced: bool = False  # 多章节问题的"每节至少一条"，不参与后续截断


def parse_docs(blocks: list[tuple[str, str, str, str]]) -> list[DocBlock]:
    """blocks: [(title, section, source, content)]"""
    return [DocBlock(t, s, src, c) for t, s, src, c in blocks]


def _strip_header(content: str) -> str:
    text = content.lstrip()
    if text.startswith("【"):
        end = text.find(_HEADER_END)
        if end > 0:
            return text[end + 1 :].lstrip("\n")
    return text


def _query_chunks(query: str) -> list[str]:
    """按功能字切出实词块：'化妆品中铅的限量是多少' → ['化妆品中铅', '限量']"""
    buf = ""
    chunks: list[str] = []
    for ch in query:
        if ch in _DELIM_CHARS or not ("\u4e00" <= ch <= "\u9fff"):
            if buf:
                chunks.append(buf)
                buf = ""
            continue
        buf += ch
    if buf:
        chunks.append(buf)
    return chunks


def _entities(query: str, glossary: Glossary | None) -> set[str]:
    """问题点名的"硬实体"：受限物质表里的词（铅、砷、汞、甲醇…）。

    刻意只认术语表里登记过的物质，而不是"切出来的单字"：
    「结果如何」会切出一个孤零零的"何"，按单字当实体会把所有资料判成不相关。
    原料级约束（卡拉胶 vs 透明质酸钠）在检索层用文件名解决，这里不管。
    """
    if not glossary:
        return set()
    return {s for s in glossary.substances if s and s in query}


@dataclass
class Concept:
    """一个"概念"：同一个意思的多种说法算一组。

    覆盖率不能按词算——中文法规写"限值"、MSDS 写 "Limit"，你不可能要求
    同一份资料里两种说法同时出现。按概念组算，"限值/Limit" 命中一个即算满足。
    """

    terms: tuple[str, ...]
    weight: float

    def hit(self, low: str) -> bool:
        return any(_hit(t, low) for t in self.terms)


def _concepts(query: str, glossary: Glossary | None) -> list[Concept]:
    concepts: list[Concept] = []
    if glossary:
        matched_zh = {zh for zh, _ in glossary.attributes if zh and zh in query}
        sibling_en = {
            e.lower() for zh, en in glossary.attributes if zh in matched_zh for e in en
        }
        for zh, en in glossary.attributes:
            if not zh:
                continue
            same = bool(sibling_en & {e.lower() for e in en})
            if zh in matched_zh or same:
                concepts.append(Concept(tuple(dict.fromkeys([zh, *en])), 2.4))

        low = query.lower()
        for item in glossary.ingredients:
            hit_zh = [a for a in item.all_zh if a and a in query]
            hit_en = [e for e in item.en if len(e) >= 4 and e.lower() in low]
            if hit_zh or hit_en:
                terms = tuple(dict.fromkeys([*item.all_zh, *item.en[:3]]))
                concepts.append(Concept(terms, 1.5))

        for name in glossary.substances:
            if name and name in query:
                concepts.append(Concept((name,), 1.8))

    for chunk in _query_chunks(query):
        if len(chunk) == 1:
            terms = (chunk,)
        elif len(chunk) <= 4:
            terms = (chunk,)
        else:
            grams = [chunk[i : i + 2] for i in range(len(chunk) - 1)]
            terms = tuple(dict.fromkeys(grams[:6]))
        concepts.append(Concept(terms, 1.0 + 0.15 * min(len(chunk), 4)))
    return concepts


def coverage(text: str, concepts: list[Concept]) -> float:
    """这段文字覆盖了问题的多少（按概念计）。"""
    if not concepts:
        return 0.0
    low = text.lower()
    total = sum(c.weight for c in concepts) or 1.0
    got = sum(c.weight for c in concepts if c.hit(low))
    return got / total


def _attr_count(query: str, glossary: Glossary | None) -> int:
    """查询里问了几个"方面"（危害、毒理、储存…）。"""
    if not glossary:
        return 0
    return sum(1 for zh, _ in glossary.attributes if zh and zh in query)


def _keywords(query: str, glossary: Glossary | None) -> dict[str, float]:
    """把查询拆成带权重的匹配词。

    权重依据：属性词（问的"什么方面"）> 成分名（问的"哪个对象"）> 普通实词。
    长度越长的实词越具体，权重越高。
    """
    weights: dict[str, float] = {}

    def add(term: str, weight: float) -> None:
        term = term.strip().lower()
        if len(term) < 1:
            return
        weights[term] = max(weights.get(term, 0.0), weight)

    if glossary:
        low = query.lower()
        matched = [(zh, en) for zh, en in glossary.attributes if zh and zh in query]
        # 同一概念的不同中文说法（限量 / 限值 / 上限）共享英文名，命中一个就把兄弟词一起
        # 加权，否则问"限量"匹配不到法规表里的"限值"
        sibling_en = {e.lower() for _, en in matched for e in en}
        for zh, en in glossary.attributes:
            same_concept = bool(sibling_en & {e.lower() for e in en})
            if not (zh and (zh in query or same_concept)):
                continue
            for term in en:
                add(term, 2.4)
            add(zh, 2.0 if zh in query else 1.8)

        for item in glossary.ingredients:
            hit_zh = [a for a in item.all_zh if a and a in query]
            hit_en = [e for e in item.en if len(e) >= 4 and e.lower() in low]
            # 查询里没提到这个原料就完全不加权，否则全表成分名都会成为"关键词"
            if not hit_zh and not hit_en:
                continue
            for term in item.all_zh:
                add(term, 1.5)
            for term in item.en[:3]:
                add(term, 1.5)

        # 单字物质名（铅、砷、汞）必须单独加：'化妆品中铅' 切出来的二元组是
        # '中铅'，不会产生单独的 '铅'，于是"铅 10"这一行永远命中不了
        for name in glossary.substances:
            if name and name in query:
                add(name, 1.8)

    for chunk in _query_chunks(query):
        if len(chunk) == 1:
            add(chunk, 1.2)
        elif len(chunk) <= 4:
            add(chunk, 1.0 + 0.15 * len(chunk))
        else:
            for size in (3, 2):
                for i in range(len(chunk) - size + 1):
                    add(chunk[i : i + size], 0.8 + 0.1 * size)

    for token in _ASCII.findall(query):
        token = token.lower().strip(".-")
        if len(token) >= 2 and token not in {"the", "and", "for", "msds"}:
            add(token, 1.5)
    return weights


def _hit(key: str, low: str) -> bool:
    """英文词要求词边界，且统一小写比较。

    不做边界判断的话，MSDS 里的 `infothaiglycerine@yahoo.com` 会因为包含
    `glycerine` 被判成命中，直接把邮箱当成"甘油的危害"的证据。
    术语表里的 "Toxicological" 是首字母大写的，不统一大小写就永远匹配不到。
    """
    if key.isascii():
        return (
            re.search(rf"(?<![a-z]){re.escape(key.lower())}(?![a-z])", low) is not None
        )
    return key in low


def _split_units(body: str) -> list[str]:
    """切出候选单元：优先按行（MSDS 大量是键值行），过长的行再按句号切。"""
    lines = [_WS.sub(" ", ln).strip() for ln in body.splitlines()]
    merged: list[str] = []
    for line in lines:
        if not line or _NOISE_LINE.match(line) or _NOISE_UNIT.search(line):
            continue
        # 「CAS No.：」这种只有标签的行，和下一行是一对，先拼起来再打分
        if merged and (merged[-1].endswith(("：", ":")) or len(merged[-1]) < 12):
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)

    units: list[str] = []
    for line in merged:
        if len(line) <= 160:
            units.append(line)
            continue
        units.extend(s for s in _SENT_SPLIT.split(line) if s.strip())
    return [u.strip() for u in units if len(u.strip()) >= 4]


def _focus(text: str, keys: set[str], width: int = 130) -> str:
    """把长句裁到关键词附近，避免引用整段 PDF 页面。"""
    if len(text) <= width:
        return text
    low = text.lower()
    positions = [low.find(k) for k in keys if k and k in low]
    if not positions:
        return text[:width] + "…"
    start = max(0, min(positions) - 35)
    end = min(len(text), start + width)
    # 尽量从词边界开始，不要把一个词切成两半
    if start > 0:
        space = text.find(" ", start)
        if 0 <= space < start + 12:
            start = space + 1
    head = "…" if start > 0 else ""
    tail = "…" if end < len(text) else ""
    return f"{head}{text[start:end].strip()}{tail}"


def _entity_value_near(text: str, keys: set[str], window: int = 14) -> bool:
    low = text.lower()
    for key in keys:
        start = low.find(key)
        while start >= 0:
            if _NUM.search(text[start + len(key) : start + len(key) + window]):
                return True
            start = low.find(key, start + 1)
    return False


def _pick(query: str, docs: list[DocBlock], glossary: Glossary | None) -> list[Unit]:
    weights = _keywords(query, glossary)
    if not weights:
        return []

    candidates: list[Unit] = []
    for doc in docs:
        body = _strip_header(doc.content)
        section_low = doc.section.lower()
        section_hit = any(_hit(k, section_low) for k in weights)
        title_norm = _WS.sub("", doc.title).lower()
        for text in _split_units(body):
            norm = _WS.sub("", text).lower()
            if _is_heading(text, doc.section):
                continue
            if _BOILERPLATE.search(text):
                continue
            # 与文件名相同的行是页眉，不是答案
            if len(norm) <= 40 and norm in title_norm:
                continue
            candidates.append(Unit(text=text, doc=doc, section_hit=section_hit))

    if not candidates:
        return []

    # 属性词是"问的哪个方面"：查询里一旦有属性词，答案行必须涉及它。
    # 否则「甘油有什么危害」会被含公司地址的行骗走——那行也有 Glycerine。
    attr_terms = {k for k, w in weights.items() if w >= 2.0}

    # 若某块资料的章节标题本身就点了属性（"3. Hazards Identification"），这一节才是
    # 权威出处；其他章节正文里顺带提到 hazards（如「5. 消防措施 · 特殊危害」）不算
    titled: set[str] = set()
    if attr_terms:
        titled = {
            d.section for d in docs if any(_hit(a, d.section.lower()) for a in attr_terms)
        }

    # 先算词频，用 IDF 压掉"化妆品"这种满篇都是的词
    df: dict[str, int] = {}
    lows = [c.text.lower() for c in candidates]
    for term in weights:
        df[term] = sum(1 for low in lows if _hit(term, low))
    total = len(candidates)
    idf = {t: 0.35 + 1.65 * (1.0 - df[t] / total) for t in weights}

    # 问"多少/CAS/分子量"时，带数字的行几乎就是答案行
    value_probe = bool(_VALUE_PROBE.search(query))
    # 中文提问优先引用中文原文：技术资料里的英文段落可以佐证，但先给能直接读懂的部分
    wants_cjk = bool(_CJK_CHAR.search(query))

    for unit in candidates:
        low = unit.text.lower()
        hits = {k for k in weights if _hit(k, low)}
        if not hits:
            # 落在正确章节里但本行没写属性词（MSDS 常见："Inhalation: Irritation not
            # expected."），这仍然是该节正文，作为弱证据保留
            if unit.section_hit:
                unit.keys = set()
                # 同一节里越完整的一行信息量越大，避免碎片（"piratory irritation…"）胜出
                unit.score = 1.0 + min(len(unit.text) / 220.0, 0.55)
            continue
        unit.keys = hits
        score = sum(weights[k] * idf[k] for k in hits)
        # 没命中属性词也不在属性章节里的行，只是顺带沾了个词，压到最低
        weak = attr_terms and not (hits & attr_terms) and not unit.section_hit
        if weak:
            score *= 0.3
        # 章节名命中说明整节都在讲这件事，行内再出现该词就更可信
        if unit.section_hit:
            score += 1.2
        if value_probe and _NUM.search(unit.text):
            score += 1.1
        if wants_cjk and _CJK_CHAR.search(unit.text):
            score += 0.8
        # 只命中泛词（如"化妆品"）而没有属性词/实体词的行，不足以当证据
        if not any(w >= 1.45 for w in (weights[k] for k in hits)):
            score *= 0.45
        # 过短的行多半是标签（"Section 3"），过长则信息稀释
        if len(unit.text) < 10:
            score -= 0.6
        unit.score = score

    scored = [u for u in candidates if u.score > 0]

    # 实体闸门：问题点了具体物质，答案就必须提到它
    ents = _entities(query, glossary)
    if ents:
        with_entity = [u for u in scored if u.keys & ents]
        if with_entity:
            scored = with_entity
        elif any(u.keys & attr_terms for u in scored) or titled:
            # 点了名的物质在候选资料里一次都没出现，说明没查到——宁可说没有，
            # 也不能拿一段讲"限量"却不含该物质的通用条文充数
            return []

    # 属性闸门：章节标题点了属性时以该节为准；否则优先保留正文直接命中属性的行。
    # 注意这里是"有才收窄"，不是"没有就清空"——法规页面没有小节标题，
    # 强行要求属性词会把 "铅 10 砷 2" 这一行直接滤掉。
    if attr_terms:
        if titled:
            keep = [u for u in scored if u.doc.section in titled]
        else:
            keep = [u for u in scored if (u.keys & attr_terms) or u.section_hit]
        if keep:
            scored = keep

    # 实体后面紧跟数字（"铅 10"、"CAS No.: 9067-32-7"）是限量/编号类问题的答案形态
    if value_probe:
        for u in scored:
            if _entity_value_near(u.text, u.keys):
                u.score += 2.2

    scored.sort(key=lambda u: u.score, reverse=True)

    picked: list[Unit] = []
    seen: set[str] = set()
    for unit in scored:
        norm = _WS.sub("", unit.text).lower()
        if not norm or norm in seen:
            continue
        # 一行是另一行的子串时只保留信息更多的那个
        if any(norm in s or s in norm for s in seen):
            continue
        seen.add(norm)
        picked.append(unit)
        if len(picked) >= 5:
            break

    # 一个问题可能同时问两件事（"危害 + 毒理"）：命中的每个章节至少给一条，
    # 否则某一节的得分高一点，另一节就被裁掉了，答一半比不答更糟
    if len(titled) > 1:
        covered = {u.doc.section for u in picked}
        for section in titled - covered:
            best = next((u for u in scored if u.doc.section == section), None)
            if best is not None and best not in picked:
                best.forced = True
                picked.append(best)
    return picked


def compose(
    query: str,
    docs: list[DocBlock],
    glossary: Glossary | None = None,
    max_units: int = 3,
    max_groups: int = 2,
    keep_ratio: float = 0.55,
) -> str:
    """生成短答：按出处分组，逐条列原文，不做任何推断。"""
    ranked = _pick(query, docs, glossary)
    if not ranked:
        return ""
    # 只保留与最佳证据同一量级的行，避免把弱相关章节（如"第1节 名称"）凑进答案
    cutoff = ranked[0].score * keep_ratio

    # 但"问了两件事"时必须两节都答：先给每个章节留一个名额，再按分数补齐
    best_per_section: dict[str, Unit] = {}
    for unit in ranked:
        best_per_section.setdefault(unit.doc.section, unit)

    # 只有"一次问了两件事"才需要每节各留一条；问一件事时多出来的章节是噪声
    if _attr_count(query, glossary) >= 2 and len(best_per_section) > 1:
        chosen = list(best_per_section.values())
        for unit in ranked:
            if len(chosen) >= max_units:
                break
            if unit not in chosen and unit.score >= cutoff:
                chosen.append(unit)
        units = sorted(chosen, key=lambda u: u.score, reverse=True)[:max_units]
    else:
        units = [u for u in ranked if u.score >= cutoff][:max_units]

    groups: dict[tuple[str, str], list[str]] = {}
    for unit in units:
        text = _focus(unit.text, unit.keys)
        bucket = groups.setdefault((unit.doc.title, unit.doc.section), [])
        if text not in bucket:
            bucket.append(text)

    parts: list[str] = []
    for (title, section), texts in list(groups.items())[:max_groups]:
        head = f"《{title}》" + (f" · {section}" if section else "")
        parts.append(head)
        parts.extend(f"· {t}" for t in texts)

    body = "\n".join(parts)
    if len(body) > 520:
        body = body[:520].rstrip() + "…"
    return f"{body}\n\n（以上为资料原文摘录，未作推断。）"
