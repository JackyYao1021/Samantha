"""Check services and installed models without downloading or loading weights."""

import importlib.util

import httpx
from openai import OpenAI
from qdrant_client import QdrantClient


def diagnose(settings):
    checks = []
    try:
        with OpenAI(base_url=settings.vl_url, api_key=settings.vl_key, max_retries=0,
                    http_client=httpx.Client(timeout=10, trust_env=False)) as client:
            available = [model.id for model in client.models.list().data]
        ready = settings.vl_model in available
        checks.append({"component": "Qwen2.5-VL", "ready": ready,
                       "detail": f"Configured: {settings.vl_model}; available: {', '.join(available)}"})
    except Exception as exc:
        checks.append({"component": "Qwen2.5-VL", "ready": False, "detail": str(exc)})
    if settings.embedding_backend == "flagembedding":
        installed = importlib.util.find_spec("FlagEmbedding") is not None
        checks.append({"component": "BGE-M3", "ready": installed,
                       "detail": "FlagEmbedding installed; weights are checked on first encode."
                       if installed else "Install work/requirements-content-hybrid.txt; weights load on first encode."})
    else:
        try:
            with httpx.Client(timeout=10, trust_env=False) as client:
                response = client.get(settings.ollama_url.rstrip("/") + "/api/tags")
                response.raise_for_status()
                available = [model["name"] for model in response.json()["models"]]
            name = settings.embedding_model
            ready = name in available or (":" not in name and name + ":latest" in available)
            checks.append({"component": "BGE-M3", "ready": ready,
                           "detail": f"Dense-only Ollama model: {name}; available: {', '.join(available)}"})
        except Exception as exc:
            checks.append({"component": "BGE-M3", "ready": False, "detail": str(exc)})
    try:
        options = {"url": settings.qdrant_url, "api_key": settings.qdrant_key or None,
                   "timeout": 10} if settings.qdrant_url else {"path": settings.qdrant_path}
        client = QdrantClient(**options)
        try:
            collections = [collection.name for collection in client.get_collections().collections]
        finally:
            client.close()
        checks.append({"component": "Qdrant", "ready": True,
                       "detail": f"Collections: {', '.join(collections) or '(empty)'}"})
    except Exception as exc:
        checks.append({"component": "Qdrant", "ready": False, "detail": str(exc)})
    return checks
