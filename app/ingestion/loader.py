"""文档解析：把 PDF / Word / PPT / Markdown 变成带结构的段落。

三类资料的切法完全不同，因此在解析阶段就识别 profile：
  regulation —— 法规，按页切，保留章节上下文
  msds       —— 原料安全数据表，按标准十六节切
  deck       —— 演示文稿，按页切，标题即章节
  page       —— 兜底，按页切
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".pptx", ".md", ".txt"}
_SKU_RE = re.compile(r"(?:SKU|货号|产品编号)\s*[:：]?\s*([A-Za-z0-9\-]{4,})")
_CAS_RE = re.compile(r"CAS\s*(?:No\.?|号)?\s*[：:.]?\s*(\d{2,7}-\d{2}-\d)")
_PAGE_NO_RE = re.compile(r"^\s*(?:[●•]\s*)?(\d{1,3})\s*$")

# MSDS 至少有四种版式：1 - / 1. / Section 1： / SECTION 1:
_MSDS_HEAD_RE = re.compile(
    r"^\s*(?:SECTION|Section|Sec\.?)?\s*(\d{1,2})\s*[.\-–—:：]?\s*"
    r"([A-Z][A-Za-z0-9 ,/&()'\-+]{3,80})\s*$",
    re.M,
)


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class RawDoc:
    source: str
    title: str
    profile: str = "page"
    sku: str = ""
    sections: list[Section] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 各类型读取


def _read_pdf_pages(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out: list[Section] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            out.append(Section(f"第 {page_no} 页", text))
    return out


def _strip_page_furniture(pages: list[str]) -> list[str]:
    """删掉每页重复出现的页眉页脚（公司地址、电话、页脚编号）。

    这些行在 MSDS 里每页都有，会占满第一个切块的配额，让真正的内容被挤到
    后面的块里——检索时命中"第 3 节 危害"，却只能看到地址和电话。
    """
    if len(pages) < 3:
        return pages

    def norm(line: str) -> str:
        return re.sub(r"\s+", "", line).lower()

    counts: dict[str, int] = {}
    for page in pages:
        for line in {norm(ln) for ln in page.splitlines() if ln.strip()}:
            counts[line] = counts.get(line, 0) + 1

    threshold = max(3, int(len(pages) * 0.5))
    repeated = {k for k, v in counts.items() if v >= threshold and len(k) <= 90}
    if not repeated:
        return pages

    cleaned: list[str] = []
    for page in pages:
        kept = [
            ln
            for ln in page.splitlines()
            if ln.strip() and norm(ln) not in repeated
        ]
        cleaned.append("\n".join(kept))
    return cleaned


def _read_msds(path: Path) -> list[Section]:
    """按标准十六节切分；版式识别失败时退回按页切。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = _strip_page_furniture([(p.extract_text() or "") for p in reader.pages])
    full = "\n".join(pages)

    marks: list[tuple[int, str]] = []
    expected = 1
    for m in _MSDS_HEAD_RE.finditer(full):
        num = int(m.group(1))
        if num == expected:
            marks.append((m.start(), f"{num}. {m.group(2).strip()}"))
            expected += 1
        if expected > 16:
            break

    if len(marks) < 4:
        return _read_pdf_pages(path)

    sections: list[Section] = []
    for idx, (pos, heading) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(full)
        body = full[pos:end].strip()
        if body:
            sections.append(Section(heading, body))
    return sections


def _load_deck_labels(path: Path) -> list[dict]:
    """同目录下的 deck_labels.yaml 提供按页段的中文主题，解决中文查询命中不了英文 PPT。"""
    sidecar = path.parent / "deck_labels.yaml"
    if not sidecar.exists():
        return []
    try:
        import yaml

        data = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
        return data.get("labels", []) or []
    except Exception as exc:
        logger.warning("页面主题标签解析失败", extra={"extra_fields": {"err": str(exc)}})
        return []


def _label_for(page_no: int, labels: list[dict]) -> str:
    for item in labels:
        span = item.get("pages") or []
        if len(span) == 2 and span[0] <= page_no <= span[1]:
            return f"{item.get('zh', '')} {item.get('en', '')}".strip()
    return ""


def _read_pptx(path: Path) -> list[Section]:
    from pptx import Presentation

    prs = Presentation(str(path))
    deck_labels = _load_deck_labels(path)
    out: list[Section] = []
    for idx, slide in enumerate(prs.slides, start=1):
        blocks: list[tuple[float, str]] = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = (shape.text_frame.text or "").strip()
                if not text or _PAGE_NO_RE.match(text):
                    continue
                top = getattr(shape, "top", None)
                blocks.append((float(top) if top is not None else 1e9, text))
            if getattr(shape, "has_table", False):
                rows = []
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    top = getattr(shape, "top", None)
                    blocks.append((float(top) if top is not None else 1e9, "\n".join(rows)))

        if not blocks:
            continue
        blocks.sort(key=lambda b: b[0])
        # 版式上最靠上的文本块通常就是该页标题，用它做章节名而不是「第 N 页」
        heading = " ".join(blocks[0][1].split())[:70]
        topic = _label_for(idx, deck_labels)
        if topic:
            heading = f"P{idx} {topic}｜{heading}"
        else:
            heading = f"P{idx} {heading}"
        body = "\n".join(dict.fromkeys(t for _, t in blocks))
        out.append(Section(heading or f"第 {idx} 页", body))
    return out


def _read_docx(path: Path) -> list[Section]:
    from docx import Document

    doc = Document(str(path))
    sections: list[Section] = []
    heading = "正文"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(p for p in buffer if p.strip())
        if body.strip():
            sections.append(Section(heading, body))
        buffer = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        if style.startswith("heading") or style.startswith("title"):
            flush()
            heading = text
        else:
            buffer.append(text)
    flush()
    return sections


def _read_plain(path: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    sections: list[Section] = []
    heading = "正文"
    buffer: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if buffer:
                sections.append(Section(heading, "\n".join(buffer)))
                buffer = []
            heading = stripped.lstrip("#").strip() or "正文"
        elif stripped:
            buffer.append(stripped)
    if buffer:
        sections.append(Section(heading, "\n".join(buffer)))
    return sections


# ---------------------------------------------------------------- profile 判定


@dataclass
class Source:
    label: str
    path: Path
    recursive: bool = True


def detect_profile(path: Path, head: str) -> str:
    if path.suffix.lower() == ".pptx":
        return "deck"
    probe = head.lower()
    if "safety data sheet" in probe or "安全技术说明书" in head:
        return "msds"
    if "化妆品安全技术规范" in head or "safety and technical standards for cosmetics" in probe:
        return "regulation"
    return "page"


_READERS = {
    ".pdf": _read_pdf_pages,
    ".docx": _read_docx,
    ".pptx": _read_pptx,
    ".md": _read_plain,
    ".txt": _read_plain,
}


# ---------------------------------------------------------------- 对外入口


def load_sources(sources: list[Source]) -> list[RawDoc]:
    """sources: 带标签的输入源；标签会写进 source 字段，避免跨目录重名。"""
    docs: list[RawDoc] = []
    for src in sources:
        root = Path(src.path)
        if not root.exists():
            logger.warning("目录不存在，已跳过", extra={"extra_fields": {"dir": str(root)}})
            continue
        files = root.rglob("*") if src.recursive else root.iterdir()
        for file in sorted(files):
            if not file.is_file() or file.suffix.lower() not in SUPPORTED:
                continue
            doc = _load_one(src.label, file)
            if doc:
                docs.append(doc)
    return docs


def _load_one(label: str, file: Path) -> RawDoc | None:
    try:
        if file.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(file))
            head = "\n".join(
                (p.extract_text() or "") for p in reader.pages[:3]
            )[:6000]
        elif file.suffix.lower() == ".pptx":
            head = ""
        else:
            head = file.read_text(encoding="utf-8", errors="ignore")[:4000]
    except Exception as exc:
        logger.error("读取失败", extra={"extra_fields": {"file": file.name, "err": str(exc)}})
        return None

    profile = detect_profile(file, head)
    reader = _read_msds if profile == "msds" else _READERS[file.suffix.lower()]
    try:
        sections = reader(file)
    except Exception as exc:
        logger.error("解析失败", extra={"extra_fields": {"file": file.name, "err": str(exc)}})
        return None
    if not sections:
        logger.warning("未提取到文本", extra={"extra_fields": {"file": file.name}})
        return None

    joined = "\n".join(s.text for s in sections[:3])
    cas = _CAS_RE.search(joined)
    sku = _SKU_RE.search(joined) or _SKU_RE.search(file.stem)
    doc = RawDoc(
        source=f"{label}/{file.name}",
        title=file.stem.replace("_", " ").strip(),
        profile=profile,
        sku=sku.group(1).upper() if sku else "",
        sections=sections,
        meta={"cas": cas.group(1) if cas else "", "label": label},
    )
    logger.info(
        "文档解析完成",
        extra={
            "extra_fields": {
                "file": file.name,
                "profile": profile,
                "sections": len(sections),
                "cas": doc.meta["cas"],
            }
        },
    )
    return doc


def load_dir(path: Path) -> list[RawDoc]:
    return load_sources([Source("default", path)])
