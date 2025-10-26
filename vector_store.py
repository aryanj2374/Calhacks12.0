"""ChromaDB-backed vector store for menu retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import datetime as dt

try:  # pragma: no cover - optional dependency
    import chromadb  # type: ignore
    from chromadb.api.models.Collection import Collection  # type: ignore
    from chromadb.utils.embedding_functions import (  # type: ignore
        DefaultEmbeddingFunction,
    )
except Exception:  # pragma: no cover - fallback when chromadb missing
    chromadb = None
    Collection = None
    DefaultEmbeddingFunction = None


class VectorStoreUnavailable(RuntimeError):
    """Raised when the vector store backend cannot be used."""


@dataclass
class QueryResult:
    """Result row from the vector store query."""

    item_id: str
    distance: float
    metadata: Dict[str, Any]


class ChromaVectorStore:
    """Wrapper around a persistent Chroma collection for menu items."""

    def __init__(self, root: Path) -> None:
        if chromadb is None or DefaultEmbeddingFunction is None:
            raise VectorStoreUnavailable(
                "ChromaDB is not available. Install `chromadb` to enable vector retrieval."
            )

        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self._client = chromadb.PersistentClient(path=str(self.root))  # type: ignore[attr-defined]
        self._embedding_function = DefaultEmbeddingFunction()
        self._collection: Collection = self._client.get_or_create_collection(
            name="menu_items",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_function,
        )

    # --------------------------------------------------------------------- #
    # Index management
    # --------------------------------------------------------------------- #

    def sync(self, items: Sequence[Any], fingerprint: str) -> None:
        """Ensure the Chroma collection matches the current dataset fingerprint."""
        state = self._load_state()
        item_count = len(items)
        if (
            state.get("fingerprint") == fingerprint
            and state.get("count") == item_count
            and self.count() == item_count
        ):
            return

        self.reset()
        self.index_items(items)
        self._save_state(
            {
                "fingerprint": fingerprint,
                "count": item_count,
                "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )

    def reset(self) -> None:
        """Remove all documents in the collection."""
        self._collection.delete(where={})

    def index_items(self, items: Sequence[Any]) -> None:
        """Upsert menu items into the Chroma collection."""
        batch_size = 128
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            for item in batch:
                ids.append(item.identifier)
                documents.append(item.doc_text)
                metadatas.append(self._serialize_metadata(item))
            if ids:
                self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def count(self) -> int:
        """Return the number of documents currently stored."""
        try:
            return int(self._collection.count())
        except Exception:
            return 0

    # --------------------------------------------------------------------- #
    # Querying
    # --------------------------------------------------------------------- #

    def query(
        self,
        text: str,
        *,
        top_n: int = 50,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        """Return the closest items to the provided query text."""
        where = self._build_where(filters)
        result = self._collection.query(
            query_texts=[text],
            n_results=top_n,
            where=where,
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        if not ids:
            return []

        rows: List[QueryResult] = []
        for idx, item_id in enumerate(ids):
            distance = float(distances[idx]) if idx < len(distances) else 0.0
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            rows.append(QueryResult(item_id=item_id, distance=distance, metadata=metadata))
        return rows

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _serialize_metadata(self, item: Any) -> Dict[str, Any]:
        return {
            "name": item.name,
            "location": item.location,
            "meal": item.meal,
            "category": item.category,
            "tags": sorted(item.tag_set),
            "dietary": list(item.dietary_choices),
        }

    @staticmethod
    def _build_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        where: Dict[str, Any] = {}
        for key, value in filters.items():
            if isinstance(value, (set, list, tuple)):
                values = [val for val in value if val]
                if values:
                    where[key] = {"$in": values}
            elif value is not None:
                where[key] = value
        return where or None

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        payload = json.dumps(state, indent=2, sort_keys=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.state_path)

