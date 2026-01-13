from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
)

try:
    import orjson
except ImportError:
    orjson = None  # type: ignore

from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos import CosmosClient

# LangGraph store base types
try:
    from langgraph.store.base import BaseStore
except ImportError:
    # Fallback if langgraph is not installed
    class BaseStore:  # type: ignore
        """Fallback BaseStore interface if langgraph is not installed."""
        supports_ttl: bool = False

if TYPE_CHECKING:
    from langgraph.store.base import (
        Item,
        SearchItem,
        TTLConfig,
    )

logger = logging.getLogger(__name__)

# Suppress Azure Cosmos DB logs
logging.getLogger('azure').setLevel(logging.WARNING)
logging.getLogger('azure.cosmos').setLevel(logging.WARNING)
logging.getLogger('azure.core').setLevel(logging.WARNING)





class CosmosIndexConfig(TypedDict, total=False):
    """Configuration for vector embeddings in Cosmos DB store.
    
    Supports both Azure Cognitive Search integration and native Cosmos DB vector search.
    """
    
    embed: Any
    """Embeddings model to use for generating vectors from text."""
    
    fields: list[str]
    """List of JSON paths to extract and embed from stored values."""
    
    dims: int
    """Dimensionality of the embedding vectors."""
    
    vector_index_type: Literal["flat", "diskANN", "quantizedFlat"]
    """Type of vector index:
    - 'flat': Brute-force search (exact)
    - 'diskANN': DiskANN-based approximate nearest neighbor
    - 'quantizedFlat': Quantized flat index for reduced memory
    """
    
    distance_type: Literal["euclidean", "cosine", "dotproduct"]
    """Distance metric for similarity search:
    - 'euclidean': L2 distance
    - 'cosine': Cosine similarity
    - 'dotproduct': Dot product (inner product)
    """
    
    quantization_byte_size: Optional[int]
    """Byte size for quantization (only for quantizedFlat). Typically 1, 2, or 4."""


class Row(TypedDict):
    """Cosmos DB document structure."""
    key: str
    value: dict[str, Any]
    namespace: str
    created_at: str
    updated_at: str
    expires_at: Optional[str]
    ttl_seconds: Optional[int]
    
C = TypeVar("C", bound=CosmosClient | AsyncCosmosClient)


class BaseCosmosStore(Generic[C]):
    """Base class for Cosmos DB store implementations."""
    
    client: C
    database_name: str
    container_name: str
    _deserializer: Callable[[bytes | str], dict[str, Any]] | None
    index_config: CosmosIndexConfig | None
    ttl_config: Optional[TTLConfig]
    
    def __init__(
        self,
        client: C,
        database_name: str,
        container_name: str,
        *,
        deserializer: Callable[[bytes | str], dict[str, Any]] | None = None,
        index: CosmosIndexConfig | None = None,
        ttl: Optional[dict[str, Any]] = None,
    ):
        """Initialize Cosmos DB store.
        
        Args:
            client: Cosmos DB client (sync or async)
            database_name: Name of the database
            container_name: Name of the container
            deserializer: Optional custom deserializer for values
            index: Vector index configuration
            ttl: TTL configuration for automatic expiration
        """
        self.client = client
        self.database_name = database_name
        self.container_name = container_name
        self._deserializer = deserializer
        self.index_config = index
        self.ttl_config = ttl
        self.embeddings = None
        
        if self.index_config:
            self.embeddings = self._ensure_embeddings(self.index_config)
    
    def _ensure_embeddings(self, config: CosmosIndexConfig) -> Any:
        """Ensure embeddings model is configured."""
        embed = config.get("embed")
        if embed is None:
            raise ValueError("Embeddings model is required in index_config")
        return embed
    
    def _namespace_to_text(self, namespace: tuple[str, ...]) -> str:
        """Convert namespace tuple to text string."""
        return ".".join(namespace)
    
    def _text_to_namespace(self, text: str) -> tuple[str, ...]:
        """Convert text string to namespace tuple."""
        return tuple(text.split("."))
    
    def _get_partition_key(self, namespace: tuple[str, ...]) -> str:
        """Get partition key from namespace.
        
        Uses first part of namespace as partition key for better distribution.
        """
        if not namespace:
            return "default"
        return namespace[0]
    
    def _serialize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Serialize item for Cosmos DB storage."""
        def convert_values(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_values(v) for v in obj]
            else:
                return obj
        
        return {key: convert_values(value) for key, value in item.items()}
    
    def _deserialize_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Deserialize item from Cosmos DB."""
        # Remove Cosmos DB internal fields
        internal_fields = ['_rid', '_self', '_etag', '_attachments', '_ts']
        return {k: v for k, v in item.items() if k not in internal_fields}
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()
    
    def _calculate_ttl_expiry(self, ttl_minutes: float) -> str:
        """Calculate expiry timestamp from TTL in minutes."""
        expiry = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
        return expiry.isoformat()
    
    def _row_to_item(self, row: dict[str, Any]) -> Item | None:
        """Convert Cosmos DB document to Item."""
        value = row.get("value", {})
        if self._deserializer and isinstance(value, (str, bytes)):
            value = self._deserializer(value)
        
        return Item(
            key=row["key"],
            value=value,
            namespace=self._text_to_namespace(row["namespace"]),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
    
    def _row_to_search_item(self, row: dict[str, Any]) -> SearchItem:
        """Convert Cosmos DB document to SearchItem with score."""
        value = row.get("value", {})
        if self._deserializer and isinstance(value, (str, bytes)):
            value = self._deserializer(value)
        
        return SearchItem(
            key=row["key"],
            value=value,
            namespace=self._text_to_namespace(row["namespace"]),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            score=row.get("score", 0.0)
        )
    
    def _get_filter_condition(
        self, key: str, op: str, value: Any
    ) -> tuple[str, list[Any]]:
        """Generate filter conditions for Cosmos DB queries."""
        if op == "$eq":
            return f"c['value']['{key}'] = @value", [value]
        elif op == "$gt":
            return f"c['value']['{key}'] > @value", [value]
        elif op == "$gte":
            return f"c['value']['{key}'] >= @value", [value]
        elif op == "$lt":
            return f"c['value']['{key}'] < @value", [value]
        elif op == "$lte":
            return f"c['value']['{key}'] <= @value", [value]
        elif op == "$ne":
            return f"c['value']['{key}'] != @value", [value]
        else:
            raise ValueError(f"Unsupported operator: {op}")
    
    def _extract_text_from_path(self, value: dict[str, Any], path: str) -> str:
        """Extract text from nested dict using dot notation path."""
        parts = path.split(".")
        current = value
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return ""
            else:
                return ""
        return str(current) if current is not None else ""

