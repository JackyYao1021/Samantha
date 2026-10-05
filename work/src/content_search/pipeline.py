"""Extract, annotate, chunk, embed, commit; then retrieve or answer."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import uuid
from interaction_log import InteractionLogError, record_operation

from .extract import discover_files, extract_sections, split_text
from .models import BGEEmbedder, QwenAnalyzer, normalize_tags
from .store import ContentStore


def content_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ContentPipeline:
    def __init__(self, settings, analyzer=None, embedder=None, store=None):
        self.settings = settings
        self.analyzer = analyzer or QwenAnalyzer(settings)
        self.embedder = embedder or BGEEmbedder(settings)
        try:
            self.store = store or ContentStore(settings, self.embedder)
        except Exception:
            self.embedder.close()
            self.analyzer.close()
            raise

    def _fingerprint(self, digest):
        settings = asdict(self.settings)
        fields = ("vl_url", "vl_model", "chunk_size", "chunk_overlap", "pdf_vision")
        data = {key: settings[key] for key in fields}
        data.update(content_hash=digest, embedding=self.embedder.identity, pipeline_version=1)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _file_summary(self, summaries):
        while len(summaries) > 1:
            summaries = [self.analyzer.annotate(
                "Summarize these source-section summaries as one document overview:\n" +
                "\n".join(summaries[start:start + 8]))["summary"]
                for start in range(0, len(summaries), 8)]
        return summaries[0]

    def index_file(self, path, force=False):
        path = Path(path).expanduser().resolve(strict=True)
        file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.normcase(str(path))))
        digest = content_hash(path)
        fingerprint = self._fingerprint(digest)
        if not force and self.store.unchanged(file_id, fingerprint):
            return {"path": str(path), "status": "unchanged"}
        records, summaries, tags = [], [], []
        for section in extract_sections(path, self.settings.pdf_vision):
            if section.image is not None:
                annotation = self.analyzer.annotate(image=section.image)
                summaries.append(annotation["summary"])
                tags.extend(annotation["tags"])
                for kind, field in (("visual_description", "description"), ("ocr_text", "ocr_text")):
                    for text in split_text(annotation[field], self.settings.chunk_size, self.settings.chunk_overlap):
                        records.append({"text": text, "page": section.page, "kind": kind,
                                        "summary": annotation["summary"]})
                records.append({"text": annotation["summary"], "page": section.page,
                                "kind": "visual_summary", "summary": annotation["summary"]})
            else:
                for text in split_text(section.text, self.settings.chunk_size, self.settings.chunk_overlap):
                    annotation = self.analyzer.annotate(text)
                    summaries.append(annotation["summary"])
                    tags.extend(annotation["tags"])
                    records.append({"text": text, "page": section.page,
                                    "kind": section.kind, "summary": annotation["summary"]})
        if not records:
            raise ValueError("No searchable content found.")
        summary = self._file_summary(summaries)
        tags = normalize_tags(tags)
        records.append({"text": summary, "page": None, "kind": "file_summary", "summary": summary})
        payloads = [{**record, "chunk": index, "path": str(path), "name": path.name,
                     "extension": path.suffix.lower(), "fingerprint": fingerprint,
                     "content_hash": digest, "file_summary": summary, "tags": tags}
                    for index, record in enumerate(records)]
        texts = [f"File: {path.name}\nTags: {', '.join(tags)}\nSummary: {p['summary']}\nContent:\n{p['text']}"
                 for p in payloads]
        embeddings = []
        for start in range(0, len(texts), self.settings.batch_size):
            embeddings.extend(self.embedder.encode(texts[start:start + self.settings.batch_size]))
        if content_hash(path) != digest:
            raise ValueError("File changed during indexing; run index again.")
        self.store.replace_file(file_id, payloads, embeddings)
        return {"path": str(path), "status": "indexed", "chunks": len(records),
                "summary": summary, "tags": tags}

    def index(self, target, force=False, progress=None):
        results = []
        for path in discover_files(target):
            if progress:
                progress(f"Indexing {path}")
            try:
                results.append(record_operation("content.index_file", self.index_file, path, force))
            except InteractionLogError:
                raise
            except Exception as exc:
                results.append({"path": str(path), "status": "failed", "error": str(exc)})
        return results

    def search(self, query, limit=10, tags=(), extension=None):
        if not query.strip():
            raise ValueError("Search query must not be empty.")
        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be between 1 and 100.")
        vector = self.embedder.encode([query])[0]
        return self.store.search(vector, limit, normalize_tags(list(tags)), extension)

    def ask(self, question, limit=5, tags=(), extension=None):
        files = self.search(question, limit, tags, extension)
        sources, budget = [], 18000
        for file in files:
            for hit in file["matches"]:
                if budget <= 0:
                    break
                text = hit["text"][:min(6000, budget)]
                sources.append({"path": file["path"], **hit, "text": text})
                budget -= len(text)
        if not sources:
            return {"answer": "No matching sources found.", "sources": []}
        return {"answer": self.analyzer.answer(question, sources), "sources": sources}

    def close(self):
        self.store.close()
        self.embedder.close()
        self.analyzer.close()
