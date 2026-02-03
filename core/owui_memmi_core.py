"""
title: MEMMI Core (Modular Ecosystem for Memory & Mind Integration)
author: Luxwarp & Gemmi
author_url: https://github.com/Luxwarp/Memmi
version: 1.0.0
description: Enhanced memory system with Native Ollama support, detailed status logging, and adaptive vector storage.
"""

import json
import copy
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    Set,
    Tuple,
)
import logging
import re
import asyncio
import pytz
import difflib
import time
import os
import hashlib
import random
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ----------------------------
# Metrics & Monitoring Imports
# ----------------------------
try:
    from prometheus_client import Counter, Histogram  # type: ignore
except ImportError:

    class _NoOpMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    Counter = Histogram = _NoOpMetric

# Define Prometheus metrics (Renamed for MEMMI)
EMBEDDING_REQUESTS = Counter(
    "memmi_embedding_requests_total", "Total number of embedding requests", ["provider"]
)
EMBEDDING_ERRORS = Counter(
    "memmi_embedding_errors_total", "Total number of embedding errors", ["provider"]
)
EMBEDDING_LATENCY = Histogram(
    "memmi_embedding_latency_seconds", "Latency of embedding generation", ["provider"]
)

RETRIEVAL_REQUESTS = Counter(
    "memmi_retrieval_requests_total", "Total number of retrieval calls", []
)
RETRIEVAL_ERRORS = Counter(
    "memmi_retrieval_errors_total", "Total number of retrieval errors", []
)
RETRIEVAL_LATENCY = Histogram(
    "memmi_retrieval_latency_seconds", "Latency of retrieval execution", []
)

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import numpy as np
import aiohttp
from aiohttp import ClientError
from pydantic import BaseModel, Field, model_validator, field_validator

# OpenWebUI Imports
try:
    from open_webui.config import DATA_DIR
except ImportError:
    from pathlib import Path

    DATA_DIR = Path("/app/backend/data")

from open_webui.models.memories import Memories
from open_webui.models.users import Users
from open_webui.main import app as webui_app

# --- Router & Mock Imports for Vector Indexing ---
try:
    from open_webui.routers.memories import add_memory, AddMemoryForm
except ImportError:
    add_memory = None
    AddMemoryForm = None

# --- Vector Database Client ---
try:
    from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
except ImportError:
    VECTOR_DB_CLIENT = None


# --- Advanced Mock Infrastructure ---
class MockConfig:
    def __init__(self):
        self.ENABLE_MEMORIES = True
        self.USER_PERMISSIONS = {"features": {"memories": True}}


class MockAppState:
    def __init__(self, embedding_function):
        self.config = MockConfig()
        self.EMBEDDING_FUNCTION = embedding_function


class MockApp:
    def __init__(self, embedding_function):
        self.state = MockAppState(embedding_function)


class MockState:
    def __init__(self, user):
        self.user = user


class MockRequest:
    def __init__(self, user, embedding_function):
        self.app = MockApp(embedding_function)
        self.state = MockState(user)
        self.user = user


# Set up logging with versioned adapter
_raw_logger = logging.getLogger("openwebui.plugins.memmi_core")
if not _raw_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    _raw_logger.addHandler(handler)
    _raw_logger.setLevel(logging.INFO)


class MemmiAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f"[MEMMI Core] {msg}", kwargs


logger = MemmiAdapter(_raw_logger, {})

# ------------------------------------------------------------------------------
# Data Models and Helper Classes
# ------------------------------------------------------------------------------


class MemoryOperation(BaseModel):
    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = []
    memory_bank: Optional[str] = None
    confidence: Optional[float] = None


class LocalAddMemoryForm(BaseModel):
    content: str


class ErrorManager:
    def __init__(self):
        self.counters: Dict[str, int] = {
            "embedding_errors": 0,
            "llm_call_errors": 0,
            "json_parse_errors": 0,
            "memory_crud_errors": 0,
        }

    def increment(self, counter_name: str):
        self.counters[counter_name] = self.counters.get(counter_name, 0) + 1

    def get_counters(self) -> Dict[str, int]:
        return self.counters


class JSONParser:
    @staticmethod
    def extract_and_parse(text: str) -> Union[List, Dict, None]:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        bracket_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(1))
            except json.JSONDecodeError:
                pass
        return None


class LRUCache:
    def __init__(self, max_size: int = 10000):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[np.ndarray]:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    async def set(self, key: str, value: np.ndarray) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
            self._cache[key] = value


# ------------------------------------------------------------------------------
# Embedding Management
# ------------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    @abstractmethod
    async def get_embedding(
        self, text: str, session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    async def get_embeddings_batch(
        self, texts: List[str], session: Optional[aiohttp.ClientSession] = None
    ) -> List[Optional[np.ndarray]]:
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer:
            try:
                logger.info(f"Loading local embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.exception(f"Failed to load local SentenceTransformer model: {e}")

    async def get_embedding(
        self, text: str, session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[np.ndarray]:
        if not self.model:
            return None
        try:
            loop = asyncio.get_running_loop()
            embedding = await loop.run_in_executor(
                None, lambda: self.model.encode(text, normalize_embeddings=True)
            )
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.exception(f"Local embedding error: {e}")
            return None

    async def get_embeddings_batch(
        self, texts: List[str], session: Optional[aiohttp.ClientSession] = None
    ) -> List[Optional[np.ndarray]]:
        if not self.model or not texts:
            return [None] * len(texts)
        try:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                ),
            )
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as e:
            logger.exception(f"Local batch embedding error: {e}")
            return [None] * len(texts)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name

    async def get_embedding(
        self, text: str, session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[np.ndarray]:
        try:
            inner_session = session if session else aiohttp.ClientSession()
            should_close = session is None
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                data = {"input": text, "model": self.model_name}
                async with inner_session.post(
                    self.api_url, json=data, headers=headers, timeout=30
                ) as response:
                    if response.status == 200:
                        res_json = await response.json()
                        if "data" in res_json and len(res_json["data"]) > 0:
                            emb = res_json["data"][0]["embedding"]
                            return np.array(emb, dtype=np.float32)
                    return None
            finally:
                if should_close:
                    await inner_session.close()
        except Exception as e:
            logger.exception(f"API embedding error: {e}")
            return None

    async def get_embeddings_batch(
        self, texts: List[str], session: Optional[aiohttp.ClientSession] = None
    ) -> List[Optional[np.ndarray]]:
        try:
            inner_session = session if session else aiohttp.ClientSession()
            should_close = session is None
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                data = {"input": texts, "model": self.model_name}
                async with inner_session.post(
                    self.api_url, json=data, headers=headers, timeout=60
                ) as response:
                    if response.status == 200:
                        res_json = await response.json()
                        if "data" in res_json:
                            results = [None] * len(texts)
                            for item in res_json["data"]:
                                idx = item.get("index")
                                embedding_data = item.get("embedding")
                                if (
                                    idx is not None
                                    and 0 <= idx < len(results)
                                    and embedding_data is not None
                                ):
                                    results[idx] = np.array(
                                        embedding_data, dtype=np.float32
                                    )
                            return results
                    return [None] * len(texts)
            finally:
                if should_close:
                    await inner_session.close()
        except Exception as e:
            logger.exception(f"API batch embedding error: {e}")
            return [None] * len(texts)


class EmbeddingManager:
    """Manages embedding generation, caching, and persistence."""

    def __init__(self, get_valves: Callable[[], Any], error_manager: ErrorManager):
        self.get_valves = get_valves
        self.error_manager = error_manager
        self.cache = LRUCache()
        self.provider: Optional[EmbeddingProvider] = None
        self._current_provider_type = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def _cleanup_lock(self, user_id: str) -> None:
        if user_id in self._locks:
            lock = self._locks[user_id]
            if not lock.locked() and (
                not hasattr(lock, "_waiters") or not lock._waiters
            ):
                del self._locks[user_id]

    async def cleanup(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _ensure_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

    def _ensure_provider(self):
        valves = self.get_valves()
        if (
            not self.provider
            or self._current_provider_type != valves.embedding_provider_type
        ):
            self._current_provider_type = valves.embedding_provider_type
            if valves.embedding_provider_type == "local":
                self.provider = LocalEmbeddingProvider(valves.embedding_model_name)
            elif valves.embedding_provider_type == "openai_compatible":
                self.provider = OpenAICompatibleEmbeddingProvider(
                    valves.embedding_api_url,
                    valves.embedding_api_key,
                    valves.embedding_model_name,
                )

    async def _get_embedding_ollama(self, text: str) -> Optional[np.ndarray]:
        """Native Ollama API Handler (Updated for /api/embed vs /api/embeddings)"""
        try:
            valves = self.get_valves()
            base_url = valves.embedding_api_url.rstrip("/")

            # --- DEBUG START ---
            print(f"[MEMMI DEBUG] Requesting Embedding for text length: {len(text)}")
            # -------------------

            # 1. Try Modern Endpoint (/api/embed) - Ollama 0.1.26+
            url = f"{base_url}/api/embed"
            payload = {
                "model": valves.embedding_model_name,
                "input": text,  # OBS: 'input', inte 'prompt'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # /api/embed returnerar 'embeddings' (plural, lista av listor)
                        if "embeddings" in data and len(data["embeddings"]) > 0:
                            # print(f"[MEMMI DEBUG] Embedding Success (Modern API)")
                            return np.array(data["embeddings"][0])

            # 2. Fallback to Legacy Endpoint (/api/embeddings) if above failed
            print(
                f"[MEMMI DEBUG] Modern /api/embed failed, trying legacy /api/embeddings..."
            )
            url = f"{base_url}/api/embeddings"
            payload = {
                "model": valves.embedding_model_name,
                "prompt": text,  # OBS: 'prompt'
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "embedding" in data:
                            print(f"[MEMMI DEBUG] Embedding Success (Legacy API)")
                            return np.array(data["embedding"])
                    else:
                        error_text = await resp.text()
                        print(
                            f"[MEMMI ERROR] Embedding API Failed: {resp.status} - {error_text}"
                        )

        except Exception as e:
            print(f"[MEMMI CRITICAL] Embedding Exception: {e}")
            import traceback

            traceback.print_exc()
            return None
        return None

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        EMBEDDING_REQUESTS.labels(self.get_valves().embedding_provider_type).inc()
        start = time.perf_counter()

        # NATIVE OLLAMA BYPASS
        if self.get_valves().embedding_provider_type == "ollama":
            return await self._get_embedding_ollama(text)

        if not self.provider:
            self._ensure_provider()
        if not self.provider:
            return None

        self._ensure_session()
        emb = await self.provider.get_embedding(text, session=self._session)

        if emb is not None:
            EMBEDDING_LATENCY.labels(self.get_valves().embedding_provider_type).observe(
                time.perf_counter() - start
            )
        else:
            self.error_manager.increment("embedding_errors")
            EMBEDDING_ERRORS.labels(self.get_valves().embedding_provider_type).inc()
        return emb

    async def get_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[np.ndarray]]:
        if not self.provider:
            self._ensure_provider()
        if not self.provider:
            return [None] * len(texts)
        self._ensure_session()
        return await self.provider.get_embeddings_batch(texts, session=self._session)

    async def store_embedding_persistent(
        self, user_id: str, memory_id: str, memory_text: str, embedding: np.ndarray
    ) -> None:
        async with self._get_lock(user_id):
            try:
                cache_dir = os.path.join(DATA_DIR, "cache", "embeddings")
                await asyncio.to_thread(os.makedirs, cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, f"{user_id}_embeddings.json")
                cache = {}
                if await asyncio.to_thread(os.path.exists, cache_file):
                    try:

                        def _load():
                            with open(cache_file, "r") as f:
                                return json.load(f)

                        cache = await asyncio.to_thread(_load)
                    except Exception:
                        pass

                embedding_list = (
                    embedding.tolist()
                    if isinstance(embedding, np.ndarray)
                    else embedding
                )
                cache[str(memory_id)] = {
                    "embedding": embedding_list,
                    "model": self.get_valves().embedding_model_name,
                    "provider": self.get_valves().embedding_provider_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                def _save_atomic(data):
                    tmp_file = cache_file + ".tmp"
                    with open(tmp_file, "w") as f:
                        json.dump(data, f)
                    os.replace(tmp_file, cache_file)

                await asyncio.to_thread(_save_atomic, cache)
            except Exception as e:
                logger.warning(f"Failed to store embedding persistent: {e}")
        self._cleanup_lock(user_id)

    async def store_embeddings_batch_persistent(
        self,
        user_id: str,
        ids: List[str],
        texts: List[str],
        embeddings: List[np.ndarray],
    ) -> None:
        if not ids:
            return
        async with self._get_lock(user_id):
            try:
                cache_dir = os.path.join(DATA_DIR, "cache", "embeddings")
                await asyncio.to_thread(os.makedirs, cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, f"{user_id}_embeddings.json")
                cache = {}
                if await asyncio.to_thread(os.path.exists, cache_file):
                    try:

                        def _load():
                            with open(cache_file, "r") as f:
                                return json.load(f)

                        cache = await asyncio.to_thread(_load)
                    except Exception:
                        pass

                for memory_id, embedding in zip(ids, embeddings, strict=True):
                    if embedding is None:
                        continue
                    embedding_list = (
                        embedding.tolist()
                        if isinstance(embedding, np.ndarray)
                        else embedding
                    )
                    cache[str(memory_id)] = {
                        "embedding": embedding_list,
                        "model": self.get_valves().embedding_model_name,
                        "provider": self.get_valves().embedding_provider_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                def _save_atomic(data):
                    tmp_file = cache_file + ".tmp"
                    with open(tmp_file, "w") as f:
                        json.dump(data, f)
                    os.replace(tmp_file, cache_file)

                await asyncio.to_thread(_save_atomic, cache)
            except Exception as e:
                logger.warning(f"Failed to store batch embeddings persistent: {e}")
        self._cleanup_lock(user_id)

    async def load_embedding_persistent(
        self, user_id: str, memory_id: str
    ) -> Optional[np.ndarray]:
        result = None
        async with self._get_lock(user_id):
            try:
                cache_dir = os.path.join(DATA_DIR, "cache", "embeddings")
                cache_file = os.path.join(cache_dir, f"{user_id}_embeddings.json")
                if not await asyncio.to_thread(os.path.exists, cache_file):
                    result = None
                else:

                    def _load():
                        with open(cache_file, "r") as f:
                            return json.load(f)

                    cache = await asyncio.to_thread(_load)
                    memory_id_str = str(memory_id)
                    if memory_id_str in cache:
                        embedding_data = cache[memory_id_str]
                        embedding = np.array(
                            embedding_data["embedding"], dtype=np.float32
                        )
                        valves = self.get_valves()
                        if embedding_data.get("model") != valves.embedding_model_name:
                            result = None
                        else:
                            result = embedding
            except Exception as e:
                logger.warning(f"Error loading embedding persistent: {e}")
                result = None
        self._cleanup_lock(user_id)
        return result


# ------------------------------------------------------------------------------
# Memory Pipeline
# ------------------------------------------------------------------------------


class MemoryPipeline:
    def __init__(
        self,
        valves: Any,
        embedding_manager: EmbeddingManager,
        error_manager: ErrorManager,
    ):
        self.valves = valves
        self.embedding_manager = embedding_manager
        self.error_manager = error_manager

    async def identify_memories(
        self,
        user_message: str,
        context_memories: List[Dict[str, Any]] = None,
        query_llm_func: Callable = None,
    ) -> List[Dict[str, Any]]:
        if not user_message:
            return []

        # Hämta grundprompten från Valves
        system_prompt = self.valves.memory_identification_prompt

        # --- LUXWARP FIX: THE SCHEMA SLAP ---
        # Vi tvingar modellen att förstå exakt vilket format vi vill ha
        schema_enforcement = """
IMPORTANT: You are a backend processor. You must output a JSON ARRAY of objects.
Do NOT output a conversational summary. Do NOT output a user profile object.
Use EXACTLY this format:
[
  {
    "operation": "NEW",
    "content": "User lives in Sweden",
    "tags": ["location", "personal"],
    "memory_bank": "Personal",
    "confidence": 1.0
  }
]
Valid operations are: NEW, UPDATE, DELETE.
"""
        # Lägg ihop allt
        now = datetime.now(timezone.utc)
        full_system_prompt = f"{system_prompt}\n{schema_enforcement}\n\nCurrent Date: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        # ------------------------------------

        user_prompt = f"User Message: {user_message}"
        if context_memories:
            user_prompt += "\n\nContext Memories:\n" + "\n".join(
                [f"- {m.get('content', '')}" for m in context_memories]
            )

        if not query_llm_func:
            return []

        try:
            # Skicka den stränga prompten
            response = await query_llm_func(full_system_prompt, user_prompt)

            print(f"[MEMMI DEBUG] LLM Extraction Output: {response}")

            if not response:
                return []

            # Försök parsa
            data = JSONParser.extract_and_parse(response)

            # Om LLM:en svarade med ett objekt {} istället för lista [], försök rädda det
            if isinstance(data, dict):
                # Om den gjorde {"user_name": ...} så är det kört, men vi loggar det
                print(
                    "[MEMMI DEBUG] Warning: LLM returned a dict instead of a list. Attempting to recover..."
                )
                # Här skulle man kunna ha logik för att rädda data, men oftast är det bättre att tvinga om via prompten
                return []

            if not isinstance(data, list):
                print(
                    f"[MEMMI DEBUG] Failed: Parsed data is not a list. Type: {type(data)}"
                )
                return []

            valid_ops = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                op = item.get("operation")
                content = item.get("content")
                confidence = item.get("confidence", 0.0)

                # Acceptera bara om formatet stämmer
                if op in ["NEW", "UPDATE"] and content:
                    if confidence >= self.valves.min_confidence_threshold:
                        valid_ops.append(item)
                elif op == "DELETE" and item.get("id"):
                    valid_ops.append(item)

            print(f"[MEMMI DEBUG] Valid operations found: {len(valid_ops)}")
            return valid_ops

        except Exception as e:
            self.error_manager.increment("llm_call_errors")
            logger.exception(f"Identify memories failed: {e}")
            return []

    # async def identify_memories(
    #     self,
    #     user_message: str,
    #     context_memories: List[Dict[str, Any]] = None,
    #     query_llm_func: Callable = None,
    # ) -> List[Dict[str, Any]]:
    #     if not user_message:
    #         return []
    #     system_prompt = self.valves.memory_identification_prompt
    #     now = datetime.now(timezone.utc)
    #     system_prompt += f"\n\nCurrent Date: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    #     user_prompt = f"User Message: {user_message}"
    #     if context_memories:
    #         user_prompt += "\n\nContext Memories:\n" + "\n".join(
    #             [f"- {m.get('content', '')}" for m in context_memories]
    #         )
    #     if not query_llm_func:
    #         return []

    #     try:
    #         response = await query_llm_func(system_prompt, user_prompt)
    #         # --- DEBUG: SE VAD DEN SÄGER ---
    #         print(f"[MEMMI DEBUG] LLM Extraction Output: {response}")
    #         # -------------------------------
    #         if not response:
    #             return []
    #         data = JSONParser.extract_and_parse(response)
    #         if not isinstance(data, list):
    #             return []
    #         valid_ops = []
    #         for item in data:
    #             if not isinstance(item, dict):
    #                 continue
    #             op = item.get("operation")
    #             content = item.get("content")
    #             confidence = item.get("confidence", 0.0)
    #             if op in ["NEW", "UPDATE"] and content:
    #                 if confidence >= self.valves.min_confidence_threshold:
    #                     valid_ops.append(item)
    #             elif op == "DELETE" and item.get("id"):
    #                 valid_ops.append(item)
    #         return valid_ops
    #     except Exception as e:
    #         self.error_manager.increment("llm_call_errors")
    #         logger.exception(f"Identify memories failed: {e}")
    #         return []

    async def get_relevant_memories(
        self, query: str, user_id: str, all_memories: List[Any]
    ) -> List[Any]:
        if not query or not all_memories:
            return []
        RETRIEVAL_REQUESTS.inc()
        start_time = time.perf_counter()

        try:
            query_embedding = await self.embedding_manager.get_embedding(query)
            if query_embedding is None:
                return []
        except Exception as e:
            RETRIEVAL_ERRORS.inc()
            logger.exception(f"Error generating query embedding: {e}")
            return []

        scored_memories = []
        mem_objects = []
        texts_to_embed = []
        ids_to_embed = []

        for mem in all_memories:
            mem_content = mem.content if hasattr(mem, "content") else mem.get("content")
            mem_id = mem.id if hasattr(mem, "id") else mem.get("id")
            if not mem_id or not mem_content:
                continue

            cached_emb = await self.embedding_manager.cache.get(mem_id)
            if cached_emb is not None:
                sim = self._cosine_similarity(query_embedding, cached_emb)
                if sim >= self.valves.vector_similarity_threshold:
                    scored_memories.append((sim, mem))
            else:
                persistent_emb = await self.embedding_manager.load_embedding_persistent(
                    user_id, mem_id
                )
                if persistent_emb is not None:
                    await self.embedding_manager.cache.set(mem_id, persistent_emb)
                    sim = self._cosine_similarity(query_embedding, persistent_emb)
                    if sim >= self.valves.vector_similarity_threshold:
                        scored_memories.append((sim, mem))
                else:
                    mem_objects.append(mem)
                    texts_to_embed.append(mem_content)
                    ids_to_embed.append(mem_id)

        if texts_to_embed:
            new_embeddings = await self.embedding_manager.get_embeddings_batch(
                texts_to_embed
            )
            for i, emb in enumerate(new_embeddings):
                if emb is not None:
                    await self.embedding_manager.cache.set(ids_to_embed[i], emb)
                    sim = self._cosine_similarity(query_embedding, emb)
                    if sim >= self.valves.vector_similarity_threshold:
                        scored_memories.append((sim, mem_objects[i]))
            if any(e is not None for e in new_embeddings):
                await self.embedding_manager.store_embeddings_batch_persistent(
                    user_id, ids_to_embed, texts_to_embed, new_embeddings
                )

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_memories = [
            mem for sim, mem in scored_memories[: self.valves.related_memories_n]
        ]
        RETRIEVAL_LATENCY.observe(time.perf_counter() - start_time)
        return top_memories

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1.shape != v2.shape:
            return 0.0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _normalize_text(self, text: str) -> str:
        normalized = re.sub(r"[^\w\s]", "", text.strip().lower())
        normalized = re.sub(r"\bs\b", "", normalized)
        normalized = re.sub(r"\b(a|an|the)\b", "", normalized)
        normalized = re.sub(
            r"\b(really|very|quite|pretty|so|totally|absolutely)\b", "", normalized
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    async def process_memory_operations(
        self,
        operations: List[Dict[str, Any]],
        user_id: str,
        skip_deduplication: bool = False,
    ) -> List[Dict[str, Any]]:
        user_obj = Users.get_user_by_id(user_id)
        success_ops = []
        for op in operations:
            try:
                kind = op.get("operation")
                content = op.get("content")

                if kind == "NEW" and content:
                    tags = op.get("tags", [])
                    bank = op.get("memory_bank", "General")
                    dedup_embedding = None
                    if self.valves.deduplicate_memories and not skip_deduplication:
                        is_dupe, dedup_embedding = await self._is_duplicate(
                            content, user_id
                        )
                        if is_dupe:
                            continue

                    tags_str = ", ".join(tags) if tags else "none"
                    confidence = op.get("confidence", 1.0)
                    final_content = f"[Tags: {tags_str}] {content} [Memory Bank: {bank}] [Confidence: {confidence:.2f}]"

                    try:
                        mem_obj = None
                        try:
                            if add_memory and (AddMemoryForm or LocalAddMemoryForm):
                                FormClass = (
                                    AddMemoryForm
                                    if AddMemoryForm
                                    else LocalAddMemoryForm
                                )
                                form = FormClass(content=final_content)

                                async def mock_embedding_function(
                                    content: str, user=None
                                ):
                                    return await self.embedding_manager.get_embedding(
                                        content
                                    )

                                req = (
                                    MockRequest(user_obj, mock_embedding_function)
                                    if "MockRequest" in globals()
                                    else None
                                )
                                if req:
                                    mem_obj = await add_memory(
                                        request=req, form_data=form, user=user_obj
                                    )
                                else:
                                    raise ImportError("MockRequest not available")
                            else:
                                raise ImportError(
                                    "Router add_memory not successfully imported"
                                )
                        except Exception as add_err:
                            mem_obj = Memories.insert_new_memory(user_id, final_content)

                        memory_id = getattr(mem_obj, "id", None)
                        success_ops.append(op)
                        if dedup_embedding is not None and memory_id:
                            await self.embedding_manager.cache.set(
                                str(memory_id), dedup_embedding
                            )
                            await self.embedding_manager.store_embedding_persistent(
                                user_id, str(memory_id), content, dedup_embedding
                            )
                    except Exception as ins_err:
                        logger.error(f"Failed to insert memory: {ins_err}")

                elif kind == "UPDATE" and op.get("id") and op.get("content"):
                    memory_id = op["id"]
                    new_content = op["content"]
                    updated_memory = Memories.update_memory_by_id_and_user_id(
                        memory_id, user_id, new_content
                    )
                    if updated_memory:
                        new_embedding = await self.embedding_manager.get_embedding(
                            new_content
                        )
                        if new_embedding is not None:
                            await self.embedding_manager.cache.set(
                                str(memory_id), new_embedding
                            )
                            await self.embedding_manager.store_embedding_persistent(
                                user_id, str(memory_id), new_content, new_embedding
                            )
                            if VECTOR_DB_CLIENT:
                                try:
                                    VECTOR_DB_CLIENT.upsert(
                                        collection_name=f"user-memory-{user_id}",
                                        items=[
                                            {
                                                "id": str(memory_id),
                                                "text": new_content,
                                                "vector": (
                                                    new_embedding.tolist()
                                                    if hasattr(new_embedding, "tolist")
                                                    else new_embedding
                                                ),
                                            }
                                        ],
                                    )
                                except Exception:
                                    pass
                        success_ops.append(op)

                elif kind == "DELETE" and op.get("id"):
                    memory_id = op["id"]
                    Memories.delete_memory_by_id(memory_id)
                    if VECTOR_DB_CLIENT:
                        try:
                            VECTOR_DB_CLIENT.delete(
                                collection_name=f"user-memory-{user_id}",
                                ids=[str(memory_id)],
                            )
                        except Exception:
                            pass
                    success_ops.append(op)

            except Exception as e:
                self.error_manager.increment("memory_crud_errors")
                logger.exception(f"Memory operation failed: {e}")
        return success_ops

    async def _is_duplicate(
        self,
        text: str,
        user_id: str,
        exclude_id: str = None,
        all_memories_override: List[Any] = None,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        if not text or not self.valves.deduplicate_memories:
            return False, None
        try:
            if all_memories_override is not None:
                all_memories = all_memories_override
            else:
                all_memories = Memories.get_memories_by_user_id(user_id)
            if not all_memories:
                return False, None

            if self.valves.use_embeddings_for_deduplication:
                new_embedding = await self.embedding_manager.get_embedding(text)
                if new_embedding is None:
                    return (
                        await self._check_text_similarity(
                            text, all_memories, exclude_id=exclude_id
                        ),
                        None,
                    )

                for i, memory in enumerate(all_memories):
                    memory_id = memory.id if hasattr(memory, "id") else memory.get("id")
                    if exclude_id and str(memory_id) == str(exclude_id):
                        continue

                    memory_content = (
                        memory.content
                        if hasattr(memory, "content")
                        else memory.get("content")
                    )
                    raw_memory_content = memory_content
                    if "[Tags:" in memory_content and "[Memory Bank:" in memory_content:
                        tags_end = memory_content.find(
                            "]", memory_content.find("[Tags:")
                        )
                        if tags_end != -1:
                            bank_start = memory_content.find("[Memory Bank:", tags_end)
                            if bank_start != -1:
                                raw_memory_content = memory_content[
                                    tags_end + 1 : bank_start
                                ].strip()

                    if self._normalize_text(text) == self._normalize_text(
                        raw_memory_content
                    ):
                        return True, new_embedding

                    content_for_embedding = raw_memory_content
                    existing_embedding = await self.embedding_manager.cache.get(
                        memory_id
                    )
                    if existing_embedding is None:
                        existing_embedding = (
                            await self.embedding_manager.load_embedding_persistent(
                                user_id, memory_id
                            )
                        )
                        if existing_embedding is not None:
                            await self.embedding_manager.cache.set(
                                memory_id, existing_embedding
                            )
                        else:
                            existing_embedding = (
                                await self.embedding_manager.get_embedding(
                                    content_for_embedding
                                )
                            )
                            if existing_embedding is not None:
                                await self.embedding_manager.cache.set(
                                    memory_id, existing_embedding
                                )
                                await self.embedding_manager.store_embedding_persistent(
                                    user_id,
                                    memory_id,
                                    content_for_embedding,
                                    existing_embedding,
                                )

                    if existing_embedding is not None:
                        similarity = self._cosine_similarity(
                            new_embedding, existing_embedding
                        )
                        if similarity >= self.valves.embedding_similarity_threshold:
                            return True, new_embedding
            else:
                return (
                    await self._check_text_similarity(
                        text, all_memories, exclude_id=exclude_id
                    ),
                    None,
                )

            return False, (
                new_embedding if self.valves.use_embeddings_for_deduplication else None
            )
        except Exception as e:
            logger.error(f"Duplicate check error: {e}")
            return False, None

    async def _check_text_similarity(
        self, text: str, all_memories: List[Any], exclude_id: str = None
    ) -> bool:
        normalized_text = self._normalize_text(text)
        for i, memory in enumerate(all_memories):
            memory_id = memory.id if hasattr(memory, "id") else memory.get("id")
            if exclude_id and str(memory_id) == str(exclude_id):
                continue
            memory_content = (
                memory.content if hasattr(memory, "content") else memory.get("content")
            )
            raw_memory_content = memory_content
            if "[Tags:" in memory_content and "[Memory Bank:" in memory_content:
                tags_end = memory_content.find("]", memory_content.find("[Tags:"))
                if tags_end != -1:
                    bank_start = memory_content.find("[Memory Bank:", tags_end)
                    if bank_start != -1:
                        raw_memory_content = memory_content[
                            tags_end + 1 : bank_start
                        ].strip()

            normalized_raw = self._normalize_text(raw_memory_content)
            similarity = difflib.SequenceMatcher(
                None, normalized_text, normalized_raw
            ).ratio()
            if similarity >= self.valves.similarity_threshold:
                return True
        return False

    async def cluster_and_summarize(
        self, user_id: str, query_llm_func: Callable
    ) -> Optional[str]:
        try:
            memories = Memories.get_memories_by_user_id(user_id)
            if (
                not memories
                or len(memories) < self.valves.summarization_min_cluster_size
            ):
                return
        except Exception:
            return

        contents = [m.content for m in memories]
        ids = [m.id for m in memories]
        embeddings = []
        uncached_indices, uncached_contents = [], []

        for i, (memory_id, content) in enumerate(zip(ids, contents, strict=True)):
            cached_embedding = await self.embedding_manager.cache.get(memory_id)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                persistent_embedding = (
                    await self.embedding_manager.load_embedding_persistent(
                        user_id, memory_id
                    )
                )
                if persistent_embedding is not None:
                    await self.embedding_manager.cache.set(
                        memory_id, persistent_embedding
                    )
                    embeddings.append(persistent_embedding)
                else:
                    embeddings.append(None)
                    uncached_indices.append(i)
                    uncached_contents.append(content)

        if uncached_contents:
            new_embeddings = await self.embedding_manager.get_embeddings_batch(
                uncached_contents
            )
            for idx, new_emb in zip(uncached_indices, new_embeddings, strict=True):
                if new_emb is not None:
                    embeddings[idx] = new_emb
                    await self.embedding_manager.cache.set(ids[idx], new_emb)
            await self.embedding_manager.store_embeddings_batch_persistent(
                user_id,
                [str(ids[idx]) for idx in uncached_indices],
                uncached_contents,
                new_embeddings,
            )

        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
        if len(valid_indices) < self.valves.summarization_min_cluster_size:
            return

        clusters = []
        visited = set()
        for i in valid_indices:
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in valid_indices:
                if j in visited:
                    continue
                if (
                    self._cosine_similarity(embeddings[i], embeddings[j])
                    >= self.valves.summarization_similarity_threshold
                ):
                    cluster.append(j)
                    visited.add(j)
            if len(cluster) >= self.valves.summarization_min_cluster_size:
                clusters.append(cluster)

        for cluster_indices in clusters:
            try:
                cluster_memories = [memories[i] for i in cluster_indices]
                cluster_text = "\n".join([f"- {m.content}" for m in cluster_memories])
                summary = await query_llm_func(
                    self.valves.summarization_memory_prompt,
                    f"Memories to summarize:\n{cluster_text}",
                )
                if summary:
                    op = {
                        "operation": "NEW",
                        "content": summary,
                        "tags": ["summary"],
                        "memory_bank": "General",
                        "confidence": 1.0,
                    }
                    success_ops = await self.process_memory_operations(
                        [op], user_id, skip_deduplication=True
                    )
                    if success_ops:
                        for m in cluster_memories:
                            try:
                                Memories.delete_memory_by_id(str(m.id))
                                if VECTOR_DB_CLIENT:
                                    VECTOR_DB_CLIENT.delete(
                                        collection_name=f"user-memory-{user_id}",
                                        ids=[str(m.id)],
                                    )
                            except Exception:
                                pass
                        return f"Consolidated {len(cluster_memories)} memories into a summary."
            except Exception:
                pass
        return None


class TaskManager:
    def __init__(self, filter_instance: Any):
        self.filter = filter_instance
        self.tasks: Set[asyncio.Task] = set()

    def start_tasks(self) -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        scavenger_task = asyncio.create_task(self._scavenge_rogue_tasks())
        self.tasks.add(scavenger_task)
        scavenger_task.add_done_callback(self.tasks.discard)
        valves = self.filter.valves
        if valves.enable_summarization_task:
            task = asyncio.create_task(self.filter._summarize_old_memories_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        if valves.enable_error_logging_task:
            task = asyncio.create_task(self.filter._log_error_counters_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        if valves.enable_vector_cleanup_task:
            task = asyncio.create_task(self.filter._cleanup_vectors_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        return True

    async def stop_tasks(self):
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def _scavenge_rogue_tasks(self):
        current_task = asyncio.current_task()
        all_tasks = asyncio.all_tasks()
        for task in all_tasks:
            if task == current_task:
                continue
            if any(
                x in repr(task)
                for x in ["_summarize_old_memories_loop", "function_adaptive_memory"]
            ):
                if task not in self.tasks:
                    task.cancel()


# ------------------------------------------------------------------------------
# Main Filter Class
# ------------------------------------------------------------------------------


class Filter:
    class Valves(BaseModel):
        # LUXWARP RECOMMENDED DEFAULTS
        embedding_provider_type: Literal["local", "openai_compatible", "ollama"] = (
            Field(default="ollama", description="Provider for embedding generation")
        )
        embedding_model_name: str = Field(
            default="nomic-embed-text", description="Embedding model name"
        )
        embedding_api_url: Optional[str] = Field(
            default="http://host.docker.internal:11434", description="Embedding API URL"
        )
        embedding_api_key: Optional[str] = Field(default=None, description="API Key")

        # Background Tasks
        enable_summarization_task: bool = Field(
            default=True, description="Enable summarization task"
        )
        summarization_interval: int = Field(
            default=3600, description="Summarization interval (1h)"
        )
        enable_error_logging_task: bool = Field(
            default=True, description="Enable error logging"
        )
        error_logging_interval: int = Field(
            default=1800, description="Error log interval"
        )
        enable_vector_cleanup_task: bool = Field(
            default=True, description="Enable vector cleanup"
        )
        vector_cleanup_interval: int = Field(
            default=7200, description="Vector cleanup interval"
        )

        # Discovery Tasks
        enable_date_update_task: bool = True
        date_update_interval: int = 3600
        enable_model_discovery_task: bool = True
        model_discovery_interval: int = 7200

        # Summarization Settings
        summarization_min_cluster_size: int = 3
        summarization_similarity_threshold: float = 0.7
        summarization_max_cluster_size: int = 8
        summarization_min_memory_age_days: int = 7
        summarization_strategy: Literal["embeddings", "tags", "hybrid"] = "hybrid"
        summarization_memory_prompt: str = (
            "Summarize these user memories into a single concise paragraph."
        )

        # Filtering & Logic
        enable_json_stripping: bool = True
        enable_fallback_regex: bool = True
        enable_short_preference_shortcut: bool = True
        short_preference_no_dedupe_length: int = 100
        preference_keywords_no_dedupe: str = "favorite,love,like,prefer,enjoy"
        blacklist_topics: Optional[str] = None
        filter_trivia: bool = True
        whitelist_keywords: Optional[str] = None

        # Memory Limits & Thresholds
        max_total_memories: int = Field(
            default=1000, description="Max memories per user"
        )
        pruning_strategy: Literal["fifo", "least_relevant"] = "fifo"
        min_memory_length: int = 8
        min_confidence_threshold: float = Field(
            default=0.65, description="Min confidence to save"
        )
        recent_messages_n: int = Field(
            default=10, description="Messages context for extraction"
        )
        save_relevance_threshold: float = 0.8
        max_injected_memory_length: int = Field(
            default=2000, description="Max extracted memory context length"
        )

        # LLM Config (Extraction)
        llm_provider_type: Literal["ollama", "openai_compatible"] = "ollama"
        llm_model_name: str = "llama3.2:3b"  # Rekommenderar en snabb modell här
        llm_api_endpoint_url: str = "http://host.docker.internal:11434/api/chat"
        llm_api_key: Optional[str] = None

        # Retrieval Config
        related_memories_n: int = Field(
            default=8, description="Number of memories to retrieve"
        )
        relevance_threshold: float = Field(
            default=0.55, description="Min relevance for retrieval"
        )
        memory_threshold: float = 0.6
        vector_similarity_threshold: float = 0.60
        llm_skip_relevance_threshold: float = 0.93
        top_n_memories: int = 3
        cache_ttl_seconds: int = 86400
        use_llm_for_relevance: bool = False
        deduplicate_memories: bool = True
        use_embeddings_for_deduplication: bool = True
        embedding_similarity_threshold: float = 0.75
        similarity_threshold: float = 0.95

        # UI & Misc
        timezone: str = "Europe/Stockholm"
        show_status: bool = True
        show_memories: bool = True
        memory_format: Literal["bullet", "paragraph", "numbered"] = "bullet"

        # Memory Categories
        enable_identity_memories: bool = True
        enable_behavior_memories: bool = True
        enable_preference_memories: bool = True
        enable_goal_memories: bool = True
        enable_relationship_memories: bool = True
        enable_possession_memories: bool = True

        # Retries
        max_retries: int = 2
        retry_delay: float = 1.0

        # --- THE REAL PROMPTS (RESTORED) ---
        memory_identification_prompt: str = """You are an automated JSON data extraction system. Your ONLY function is to identify user-specific facts, preferences, goals, and personal details from the User Message.

Output valid JSON only. No conversational text. No markdown blocks.

### INSTRUCTIONS:
1. Analyze the User Message (and Context Memories if provided).
2. Extract new facts about the user.
3. Identify updates to existing memories (if content contradicts or refines old memories).
4. Identify deletions (only if user explicitly says "forget that" or "that's wrong").
5. Ignore general conversation ("Hi", "How are you"), questions about the AI, or temporary states ("I am hungry").
6. Focus on long-term facts: Names, location, hobbies, job, relationships, preferences, specific goals.

### FORMAT:
Return a JSON LIST of objects. Each object must have:
- "operation": "NEW", "UPDATE", or "DELETE"
- "content": A concise, third-person fact (e.g., "User lives in Sweden").
- "confidence": Float 0.0 to 1.0 (only include > 0.6).
- "tags": Array of strings (e.g., ["location", "personal"]).
- "memory_bank": String ("General", "Personal", or "Work").
- "id": (Only for UPDATE/DELETE) The ID of the memory to modify.

### EXAMPLES:
Input: "My name is Mike and I love Python."
Output:
[
  {"operation": "NEW", "content": "User's name is Mike", "tags": ["identity"], "memory_bank": "Personal", "confidence": 1.0},
  {"operation": "NEW", "content": "User loves Python programming", "tags": ["skill", "preference"], "memory_bank": "Work", "confidence": 0.9}
]

Input: "I actually moved to Berlin now."
Context: "- User lives in Stockholm (ID: 123)"
Output:
[
  {"operation": "UPDATE", "id": "123", "content": "User lives in Berlin", "tags": ["location"], "memory_bank": "Personal", "confidence": 1.0}
]
"""

        memory_relevance_prompt: str = """You are a memory retrieval assistant.
Your task is to rate the relevance of specific memories to a new User Message.
Return a JSON object: {"relevance_score": 0.0 to 1.0} for each memory provided.
1.0 = Critical context (Direct answer, essential fact).
0.0 = Irrelevant (Unrelated topic).
"""

        memory_merge_prompt: str = """You are a memory consolidation assistant.
Your task is to merge similar memories into a single, concise fact.
Input: A list of memories.
Output: A JSON list of strings (merged memories).
If memories conflict, keep the most recent detail.
If memories are distinct, keep them separate.
"""

        allowed_memory_banks: List[str] = ["General", "Personal", "Work"]
        default_memory_bank: str = "General"
        enable_error_counter_guard: bool = True
        error_guard_threshold: int = 5
        error_guard_window_seconds: int = 600
        debug_error_counter_logs: bool = False

    class UserValves(BaseModel):
        enabled: bool = True
        show_status: bool = True
        timezone: str = ""

    def __init__(self):
        logger.info("Initializing MEMMI Core v1.0")
        self.valves = self.Valves()
        self.error_manager = ErrorManager()
        self.embedding_manager = EmbeddingManager(
            lambda: self.valves, self.error_manager
        )
        self.task_manager = TaskManager(self)
        self._processed_messages = set()
        self._last_body = {}
        self.memory_embeddings = {}
        self.seen_users = set()
        self.notification_queue = []
        self._tasks_started = False
        self._valve_hash = None

    def _check_and_handle_valve_changes(self):
        valve_str = f"{self.valves.enable_summarization_task}_{self.valves.summarization_interval}"
        new_hash = hashlib.md5(valve_str.encode()).hexdigest()
        if self._valve_hash is None:
            self._valve_hash = new_hash
            return False
        if new_hash != self._valve_hash:
            self._valve_hash = new_hash
            if self._tasks_started:
                if (
                    hasattr(self, "_restart_task")
                    and self._restart_task
                    and not self._restart_task.done()
                ):
                    self._restart_task.cancel()
                self._restart_task = asyncio.create_task(self._restart_tasks())
            return True
        return False

    async def _restart_tasks(self):
        await self.task_manager.stop_tasks()
        self._tasks_started = False
        self.task_manager.start_tasks()
        self._tasks_started = True

    # --------------------------------------------------------------------------
    # Helper: LLM Query Wrapper (LUXWARP DEBUG EDITION)
    # --------------------------------------------------------------------------
    async def _query_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Unified LLM query method with LOUD DEBUGGING and JSON MODE."""
        valves = self.valves

        print(
            f"[MEMMI DEBUG] Attempting to extract memory using model: {valves.llm_model_name}"
        )

        # --- VIKTIGT: Lägg till instruktion om JSON i system prompten också ---
        enhanced_system_prompt = (
            system_prompt
            + "\n\nIMPORTANT: You must output ONLY valid JSON. No conversational text."
        )

        for attempt in range(valves.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    url = valves.llm_api_endpoint_url
                    headers = {"Content-Type": "application/json"}
                    if valves.llm_api_key:
                        headers["Authorization"] = f"Bearer {valves.llm_api_key}"

                    payload = {
                        "model": valves.llm_model_name,
                        "messages": [
                            {"role": "system", "content": enhanced_system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "stream": False,
                        "format": "json",
                    }

                    async with session.post(
                        url, json=payload, headers=headers, timeout=30
                    ) as resp:
                        print(f"[MEMMI DEBUG] Ollama Status Code: {resp.status}")

                        if resp.status == 200:
                            data = await resp.json()
                            if "message" in data:
                                content = data["message"]["content"]
                                print(
                                    f"[MEMMI DEBUG] LLM Extraction Output: {content[:100]}..."
                                )  # Logga början av svaret
                                return content
                            elif "choices" in data:  # OpenAI style fallback
                                content = data["choices"][0]["message"]["content"]
                                return content
                        else:
                            error_text = await resp.text()
                            print(f"[MEMMI ERROR] Ollama Response: {error_text}")
                            return None

            except Exception as e:
                print(f"[MEMMI CRITICAL] Connection failed: {e}")
                if attempt < valves.max_retries:
                    await asyncio.sleep(valves.retry_delay)

        return None

    # TODO: Delete or fix later.
    # async def _query_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
    #     valves = self.valves
    #     for attempt in range(valves.max_retries + 1):
    #         try:
    #             async with aiohttp.ClientSession() as session:
    #                 url = valves.llm_api_endpoint_url
    #                 headers = {"Content-Type": "application/json"}
    #                 if valves.llm_api_key:
    #                     headers["Authorization"] = f"Bearer {valves.llm_api_key}"
    #                 payload = {
    #                     "model": valves.llm_model_name,
    #                     "messages": [
    #                         {"role": "system", "content": system_prompt},
    #                         {"role": "user", "content": user_prompt},
    #                     ],
    #                     "stream": False,
    #                 }
    #                 async with session.post(
    #                     url, json=payload, headers=headers, timeout=30
    #                 ) as resp:
    #                     if resp.status == 200:
    #                         data = await resp.json()
    #                         if "choices" in data:
    #                             return data["choices"][0]["message"]["content"]
    #                         elif "message" in data:
    #                             return data["message"]["content"]
    #         except Exception as e:
    #             if attempt < valves.max_retries:
    #                 await asyncio.sleep(valves.retry_delay)
    #             else:
    #                 self.error_manager.increment("llm_call_errors")
    #     return None

    async def inlet(
        self, body: Dict[str, Any], __event_emitter__=None, __user__=None
    ) -> Dict[str, Any]:
        if not __user__ or not body.get("messages"):
            return body
        raw_valves = __user__.get("valves", {})
        user_valves = (
            self.UserValves(**raw_valves)
            if isinstance(raw_valves, dict)
            else self.UserValves()
        )
        if not user_valves.enabled:
            return body

        if not self._tasks_started:
            self.task_manager.start_tasks()
            self._tasks_started = True
        self._check_and_handle_valve_changes()

        user_id = __user__["id"]
        self.seen_users.add(user_id)
        messages = body["messages"]
        last_message = messages[-1]["content"]
        if isinstance(last_message, list):
            last_message = " ".join(
                [m.get("text", "") for m in last_message if m.get("type") == "text"]
            ).strip()
        if not last_message or last_message.startswith("/"):
            return body

        pipeline = MemoryPipeline(
            self.valves, self.embedding_manager, self.error_manager
        )
        try:
            all_memories = Memories.get_memories_by_user_id(user_id)
        except Exception:
            all_memories = []

        relevant_memories = []
        if all_memories:
            relevant_memories = await pipeline.get_relevant_memories(
                last_message, user_id, all_memories
            )

        if relevant_memories:
            context_text = "User Memories:\n" + "\n".join(
                [f"- {m.content}" for m in relevant_memories]
            )
            if messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{context_text}"
            else:
                messages.insert(0, {"role": "system", "content": context_text})

            if user_valves.show_status:
                count = len(relevant_memories)
                if count > 0:
                    status_dict = {
                        "type": "status",
                        "data": {
                            "description": f"🧠 Recalled {count} memories.",
                            "done": True,
                        },
                    }
                    if __event_emitter__:
                        await __event_emitter__(status_dict)
        return body

    async def outlet(
        self, body: Dict[str, Any], __event_emitter__=None, __user__=None
    ) -> Dict[str, Any]:
        if not __user__ or not body.get("messages"):
            return body
        raw_valves = __user__.get("valves", {})
        user_valves = (
            self.UserValves(**raw_valves)
            if isinstance(raw_valves, dict)
            else self.UserValves()
        )
        if not user_valves.enabled:
            return body

        user_id = __user__["id"]
        messages = body["messages"]
        user_message = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_message = m["content"]
                break

        pipeline = MemoryPipeline(
            self.valves, self.embedding_manager, self.error_manager
        )
        if user_message:
            ops = await pipeline.identify_memories(user_message, [], self._query_llm)
            success_ops = []
            if ops:
                success_ops = await pipeline.process_memory_operations(ops, user_id)

            # MEMMI NATIVE LIST HACK
            if user_valves.show_status and __event_emitter__ and success_ops:
                for op in success_ops:
                    content = op.get("content", "")
                    bank = op.get("memory_bank", "General")
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"📝 {content} ({bank})",
                                "done": True,
                            },
                        }
                    )

                count = len(success_ops)
                final_msg = f"🧠 Saved {count} memories"
                await __event_emitter__(
                    {"type": "status", "data": {"description": final_msg, "done": True}}
                )

        return body

    async def _summarize_old_memories_loop(self):
        while True:
            try:
                await asyncio.sleep(self.valves.summarization_interval)
                if self.valves.enable_summarization_task and self.seen_users:
                    pipeline = MemoryPipeline(
                        self.valves, self.embedding_manager, self.error_manager
                    )
                    active_users = list(self.seen_users)
                    for user_id in active_users:
                        await pipeline.cluster_and_summarize(user_id, self._query_llm)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(60)

    async def _log_error_counters_loop(self):
        try:
            while True:
                await asyncio.sleep(self.valves.error_logging_interval)
                logger.debug(f"Error Counters: {self.error_manager.get_counters()}")
        except asyncio.CancelledError:
            pass

    async def cleanup_orphaned_vectors(self, user_id: str):
        if not VECTOR_DB_CLIENT:
            return
        try:
            db_memories = Memories.get_memories_by_user_id(user_id)
            valid_ids = {str(m.id) for m in db_memories}
            collection_name = f"user-memory-{user_id}"
            result = VECTOR_DB_CLIENT.get(collection_name=collection_name)
            if result and "ids" in result:
                vector_ids = result["ids"]
                orphaned_ids = [vid for vid in vector_ids if vid not in valid_ids]
                if orphaned_ids:
                    VECTOR_DB_CLIENT.delete(
                        collection_name=collection_name, ids=orphaned_ids
                    )
                    return {"orphans_deleted": len(orphaned_ids)}
        except Exception:
            pass
        return {"orphans_deleted": 0}

    async def _cleanup_vectors_loop(self):
        while True:
            try:
                await asyncio.sleep(self.valves.vector_cleanup_interval)
                if self.valves.enable_vector_cleanup_task and self.seen_users:
                    active_users = list(self.seen_users)
                    for user_id in active_users:
                        await self.cleanup_orphaned_vectors(user_id)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(60)
