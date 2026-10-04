"""Named dense/sparse vectors, staged updates, and file-level result grouping."""

import hashlib
import uuid

from qdrant_client import QdrantClient, models


def match(key, value):
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def sparse_vector(values):
    return models.SparseVector(indices=list(values), values=list(values.values()))


class ContentStore:
    def __init__(self, settings, embedder, client=None):
        self.settings = settings
        self.embedder = embedder
        self.collection = settings.collection
        self.dense_name = "dense_" + hashlib.sha256(embedder.identity.encode()).hexdigest()[:16]
        self.client = client or (
            QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_key or None,
                         timeout=int(settings.timeout)) if settings.qdrant_url else
            QdrantClient(path=settings.qdrant_path)
        )
        try:
            self._ensure_collection()
        except Exception:
            if client is None:
                self.client.close()
            raise

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config={self.dense_name: models.VectorParams(
                    size=self.embedder.size, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams()} if self.embedder.hybrid else None,
            )
        config = self.client.get_collection(self.collection).config.params
        vectors = config.vectors
        if (not isinstance(vectors, dict) or self.dense_name not in vectors
                or vectors[self.dense_name].size != self.embedder.size
                or vectors[self.dense_name].distance != models.Distance.COSINE
                or (self.embedder.hybrid and "sparse" not in (config.sparse_vectors or {}))):
            raise ValueError("Collection uses a different embedding model/schema. Set CONTENT_COLLECTION to a new name.")
        if self.settings.qdrant_url:
            for field, schema in {"file_id": models.PayloadSchemaType.KEYWORD,
                                  "revision": models.PayloadSchemaType.KEYWORD,
                                  "tags": models.PayloadSchemaType.KEYWORD,
                                  "extension": models.PayloadSchemaType.KEYWORD,
                                  "kind": models.PayloadSchemaType.KEYWORD,
                                  "ready": models.PayloadSchemaType.BOOL}.items():
                self.client.create_payload_index(self.collection, field, schema, wait=True)

    def unchanged(self, file_id, fingerprint):
        points, _ = self.client.scroll(self.collection, scroll_filter=models.Filter(must=[
            match("file_id", file_id), match("fingerprint", fingerprint),
            match("kind", "file_summary"), match("ready", True),
        ]), limit=1, with_vectors=False)
        return bool(points)

    def replace_file(self, file_id, payloads, embeddings):
        if not payloads or len(payloads) != len(embeddings):
            raise ValueError("Every non-empty payload needs an embedding.")
        revision = uuid.uuid4().hex
        points = []
        for index, (payload, embedding) in enumerate(zip(payloads, embeddings)):
            embedding.validate(self.embedder.size)
            vector = {self.dense_name: embedding.dense}
            if self.embedder.hybrid:
                if embedding.sparse is None:
                    raise ValueError("Hybrid indexing requires BGE-M3 sparse output.")
                vector["sparse"] = sparse_vector(embedding.sparse)
            points.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_id}:{revision}:{index}")),
                vector=vector, payload={**payload, "file_id": file_id, "revision": revision, "ready": False},
            ))
        # Incomplete revisions cannot enter searches. Publish only after every upload succeeds.
        for start in range(0, len(points), self.settings.batch_size):
            self.client.upsert(self.collection, points[start:start + self.settings.batch_size], wait=True)
        revision_filter = models.Filter(must=[match("file_id", file_id), match("revision", revision)])
        self.client.set_payload(self.collection, {"ready": True}, points=revision_filter, wait=True)
        self.client.delete(self.collection, points_selector=models.Filter(
            must=[match("file_id", file_id)], must_not=[match("revision", revision)]), wait=True)

    def search(self, embedding, limit=10, tags=(), extension=None):
        embedding.validate(self.embedder.size)
        conditions = [match("ready", True)] + [match("tags", tag) for tag in tags]
        if extension:
            conditions.append(match("extension", "." + extension.lower().lstrip(".")))
        query_filter = models.Filter(must=conditions)
        candidates = max(100, limit * 20)
        while True:
            options = {"query": embedding.dense, "using": self.dense_name}
            if self.embedder.hybrid and embedding.sparse:
                options = {"query": models.FusionQuery(fusion=models.Fusion.RRF), "prefetch": [
                    models.Prefetch(query=embedding.dense, using=self.dense_name,
                                    filter=query_filter, limit=candidates),
                    models.Prefetch(query=sparse_vector(embedding.sparse), using="sparse",
                                    filter=query_filter, limit=candidates),
                ]}
            # Client 1.15.1's local grouped query drops prefetch filters. Group
            # filtered Query API results here so local and server modes agree.
            result = self.client.query_points(
                self.collection, query_filter=query_filter, limit=candidates,
                with_payload=True, **options,
            )
            groups = {}
            for hit in result.points:
                groups.setdefault(hit.payload["file_id"], []).append(hit)
            if len(groups) >= limit or len(result.points) < candidates or candidates >= 2000:
                break
            candidates = min(2000, candidates * 2)
        files = []
        for file_id, hits in list(groups.items())[:limit]:
            first = hits[0].payload
            files.append({
                "file_id": file_id, "path": first["path"], "name": first["name"],
                "summary": first["file_summary"], "tags": first["tags"],
                "score": hits[0].score,
                "matches": [{"page": hit.payload["page"], "kind": hit.payload["kind"],
                             "chunk": hit.payload["chunk"], "text": hit.payload["text"],
                             "score": hit.score} for hit in hits[:3]],
            })
        return files

    def close(self):
        self.client.close()
