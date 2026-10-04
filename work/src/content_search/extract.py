"""Preserve native text; use VL for scans, figures, and images."""

from dataclasses import dataclass
from pathlib import Path


TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".log", ".html"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | {".pdf", ".docx"}
EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".content-index"}


@dataclass
class Section:
    text: str
    kind: str
    page: int | None = None
    image: bytes | None = None


def discover_files(target):
    target = Path(target).expanduser().resolve(strict=True)
    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {target.suffix}")
        yield target
        return
    for path in sorted(target.rglob("*")):
        relative = path.relative_to(target)
        if any(part.startswith(".") or part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path.resolve()


def split_text(text, size=1200, overlap=150):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size.")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary >= 0:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def extract_sections(path, pdf_vision="auto"):
    extension = path.suffix.lower()
    if extension in TEXT_EXTENSIONS:
        data = path.read_bytes()
        # UTF-16 BOM must be checked before the legacy Chinese encoding fallback.
        encodings = ("utf-16",) if data.startswith((b"\xff\xfe", b"\xfe\xff")) else ("utf-8-sig", "gb18030")
        for encoding in encodings:
            try:
                text = data.decode(encoding)
                break
            except UnicodeDecodeError:
                pass
        else:
            raise ValueError(f"Cannot decode {path.name}; convert it to UTF-8.")
        if "\x00" in text:
            raise ValueError(f"Binary content in text file: {path.name}")
        yield Section(text, "original_text")
    elif extension == ".docx":
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        document = Document(path)
        blocks = []
        for block in document.iter_inner_content():
            if isinstance(block, Paragraph):
                blocks.append(block.text)
            elif isinstance(block, Table):
                blocks.extend("\t".join(cell.text for cell in row.cells) for row in block.rows)
        yield Section("\n".join(blocks), "original_text")
        for relation in document.part.rels.values():
            if relation.reltype.endswith("/image") and not relation.is_external:
                yield Section("", "visual", image=relation.target_part.blob)
    elif extension == ".pdf":
        import pymupdf
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError(f"Password-protected PDF: {path.name}")
            for number, page in enumerate(document, 1):
                text = page.get_text(sort=True).strip()
                if text:
                    yield Section(text, "original_text", number)
                visual = pdf_vision == "always" or (
                    pdf_vision == "auto" and (not text or page.get_images() or page.get_drawings())
                )
                if visual:
                    scale = min(2, 2048 / max(page.rect.width, page.rect.height))
                    image = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
                    yield Section("", "visual", number, image)
                elif not text:
                    raise ValueError(f"Page {number} has no text; enable PDF vision to read scans.")
    elif extension in IMAGE_EXTENSIONS:
        from PIL import Image
        from io import BytesIO
        with Image.open(path) as image:
            for number in range(getattr(image, "n_frames", 1)):
                image.seek(number)
                output = BytesIO()
                image.convert("RGB").save(output, format="PNG")
                yield Section("", "visual", number + 1, output.getvalue())
    else:
        raise ValueError(f"Unsupported file type: {extension}")
