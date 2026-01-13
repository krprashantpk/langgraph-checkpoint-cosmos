from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    cast,
)

try:
    import orjson
except ImportError:
    orjson = None  # type: ignore

from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos import exceptions as cosmos_exceptions
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential

# LangGraph store base types
try:
    from langgraph.store.base import BaseStore
except ImportError:
    # Fallback if langgraph is not installed
    class BaseStore:  # type: ignore
        """Fallback BaseStore interface if langgraph is not installed."""
        supports_ttl: bool = False


from langgraph.store.base import (
        GetOp,
        IndexConfig,
        Item,
        ListNamespacesOp,
        Op,
        PutOp,
        Result,
        SearchItem,
        SearchOp,
        TTLConfig)

from .base import BaseCosmosStore, CosmosIndexConfig



logger = logging.getLogger(__name__)

class CosmosStore(BaseCosmosStore[CosmosClient]):
    """Synchronous Cosmos DB-backed store with optional vector search.
    
    Example:
        ```python
        from azure.cosmos import CosmosClient
        from store.cosmos.base import CosmosStore, CosmosIndexConfig
        
        # Create client
        client = CosmosClient(
            url="https://your-account.documents.azure.com:443/",
            credential="your-key"
        )
        
        # Without vector search
        store = CosmosStore(
            client=client,
            database_name="my_database",
            container_name="my_container"
        )
        store.setup()
        
        # Store documents
        store.put(("docs",), "doc1", {"text": "Python tutorial"})
        store.put(("docs",), "doc2", {"text": "TypeScript guide"})
        
        # Get document
        doc = store.get(("docs",), "doc1")
        
        # Search (without vector search, returns by recency)
        results = store.search(("docs",), limit=10)
        
        # With vector search
        from langchain_openai import OpenAIEmbeddings
        
        index_config = CosmosIndexConfig(
            embed=OpenAIEmbeddings(),
            fields=["text"],
            dims=1536,
            vector_index_type="diskANN",
            distance_type="cosine"
        )
        
        store_with_vectors = CosmosStore(
            client=client,
            database_name="my_database",
            container_name="my_container",
            index=index_config
        )
        store_with_vectors.setup()
        
        # Store with indexing
        store_with_vectors.put(("docs",), "doc1", {"text": "Python tutorial"})
        
        # Semantic search
        results = store_with_vectors.search(("docs",), query="programming guides", limit=2)
        ```
    
    Note:
        Make sure to call `setup()` before first use to create/configure the container.
        
    Warning:
        If you provide a TTL configuration, you must explicitly call `start_ttl_sweeper()`
        to begin the background thread that removes expired items.
    """
    
    __slots__ = (
        "database",
        "container",
        "_ttl_sweeper_thread",
        "_ttl_stop_event",
    )
    
    supports_ttl: bool = True
    
    def __init__(
        self,
        client: CosmosClient,
        database_name: str,
        container_name: str,
        *,
        deserializer: Callable[[bytes | str], dict[str, Any]] | None = None,
        index: CosmosIndexConfig | None = None,
        ttl: Optional[dict[str, Any]] = None,
    ):
        """Initialize synchronous Cosmos DB store."""
        super().__init__(
            client=client,
            database_name=database_name,
            container_name=container_name,
            deserializer=deserializer,
            index=index,
            ttl=ttl,
        )
        self.database = self.client.get_database_client(database_name)
        self.container = self.database.get_container_client(container_name)
        self._ttl_sweeper_thread: Optional[threading.Thread] = None
        self._ttl_stop_event = threading.Event()
    
    @classmethod
    @contextmanager
    def from_connection_string(
        cls,
        connection_string: str,
        database_name: str,
        container_name: str,
        *,
        index: CosmosIndexConfig | None = None,
        ttl: Optional[dict[str, Any]] = None,
    ) -> Iterator[CosmosStore]:
        """Create a new CosmosStore from a connection string.
        
        Args:
            connection_string: Cosmos DB connection string
            database_name: Name of the database
            container_name: Name of the container
            index: Vector index configuration
            ttl: TTL configuration
            
        Returns:
            CosmosStore instance
        """
        client = CosmosClient.from_connection_string(connection_string)
        try:
            yield cls(
                client=client,
                database_name=database_name,
                container_name=container_name,
                index=index,
                ttl=ttl,
            )
        finally:
            client.close()
    
    def setup(self) -> None:
        """Set up the Cosmos DB container.
        
        Creates the database and container if they don't exist.
        Configures vector indexing if index_config is provided.
        """
        # Create database if it doesn't exist
        try:
            self.client.create_database(self.database_name)
            logger.info(f"Created database: {self.database_name}")
        except cosmos_exceptions.CosmosResourceExistsError:
            logger.debug(f"Database already exists: {self.database_name}")
        
        # Configure indexing policy
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": '/"_etag"/?'}],
        }
        
        # Configure vector indexes if needed
        vector_embedding_policy = None
        if self.index_config:
            dims = self.index_config.get("dims", 1536)
            distance_type = self.index_config.get("distance_type", "cosine")
            vector_index_type = self.index_config.get("vector_index_type", "diskANN")
            
            vector_embedding_policy = {
                "vectorEmbeddings": [
                    {
                        "path": "/embedding",
                        "dataType": "float32",
                        "dimensions": dims,
                        "distanceFunction": distance_type,
                    }
                ]
            }
            
            # Add vector index to indexing policy
            if "vectorIndexes" not in indexing_policy:
                indexing_policy["vectorIndexes"] = []
            
            indexing_policy["vectorIndexes"].append({
                "path": "/embedding",
                "type": vector_index_type,
            })
        
        # Create container with partition key and indexing policy
        try:
            container_definition = {
                "id": self.container_name,
                "partition_key": PartitionKey(path="/partition_key"),
                "indexing_policy": indexing_policy,
            }
            
            if vector_embedding_policy:
                container_definition["vector_embedding_policy"] = vector_embedding_policy
            
            self.database.create_container(**container_definition)
            logger.info(f"Created container: {self.container_name}")
        except cosmos_exceptions.CosmosResourceExistsError:
            logger.debug(f"Container already exists: {self.container_name}")
    
    def get(
        self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool = False
    ) -> Item | None:
        """Get a single item from the store.
        
        Args:
            namespace: Namespace tuple
            key: Item key
            refresh_ttl: Whether to refresh TTL on access
            
        Returns:
            Item dict or None if not found
        """
        namespace_str = self._namespace_to_text(namespace)
        partition_key = self._get_partition_key(namespace)
        doc_id = f"{namespace_str}__{key}"
        
        try:
            item = self.container.read_item(
                item=doc_id,
                partition_key=partition_key
            )
            
            # Refresh TTL if requested
            if refresh_ttl and item.get("ttl_seconds"):
                item["expires_at"] = self._calculate_ttl_expiry(
                    item["ttl_seconds"] / 60
                )
                item["updated_at"] = self._get_current_timestamp()
                self.container.upsert_item(self._serialize_item(item))
            
            return self._row_to_item(item)
            
        except cosmos_exceptions.CosmosResourceNotFoundError:
            return None
    
    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        *,
        index: Optional[bool | list[str]] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Store an item in the store.
        
        Args:
            namespace: Namespace tuple
            key: Item key
            value: Item value (must be JSON-serializable dict)
            index: Whether/what to index for vector search
            ttl: Time-to-live in minutes (optional)
        """
        namespace_str = self._namespace_to_text(namespace)
        partition_key = self._get_partition_key(namespace)
        doc_id = f"{namespace_str}__{key}"
        timestamp = self._get_current_timestamp()
        
        item = {
            "id": doc_id,
            "partition_key": partition_key,
            "namespace": namespace_str,
            "key": key,
            "value": value,
            "created_at": timestamp,
            "updated_at": timestamp,
            "expires_at": None,
            "ttl_seconds": None,
        }
        
        # Handle TTL
        if ttl is not None:
            ttl_seconds = int(ttl * 60)
            item["ttl_seconds"] = ttl_seconds
            item["expires_at"] = self._calculate_ttl_expiry(ttl)
        
        # Handle vector embeddings
        if self.index_config and self.embeddings and index is not False:
            if index is None:
                # Use default fields from config
                fields = self.index_config.get("fields", [])
            else:
                # Use specified fields
                fields = index if isinstance(index, list) else self.index_config.get("fields", [])
            
            # Extract text from specified fields
            texts_to_embed = []
            for field in fields:
                text = self._extract_text_from_path(value, field)
                if text:
                    texts_to_embed.append(text)
            
            if texts_to_embed:
                # Generate embeddings
                # Concatenate texts with space separator
                combined_text = "\n\n".join(texts_to_embed)
                # Generate embedding for combined text
                embeddings = self.embeddings.embed_documents([combined_text])
                item["embedding"] = embeddings[0]
        
        self.container.upsert_item(self._serialize_item(item))
    
    def search(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool = False,
    ) -> list[SearchItem]:
        """Search for items in the store.
        
        Args:
            namespace_prefix: Namespace prefix to search within
            query: Text query for semantic search (requires index_config)
            filter: Filter conditions
            limit: Maximum number of results
            offset: Offset for pagination
            refresh_ttl: Whether to refresh TTL on access
            
        Returns:
            List of SearchItem dicts with scores
        """
        namespace_str = self._namespace_to_text(namespace_prefix)
        
        # Build query
        query_parts = ["SELECT * FROM c"]
        query_params: list[dict[str, Any]] = []
        conditions = []
        
        # Namespace filter
        if namespace_prefix:
            partition_key = self._get_partition_key(namespace_prefix)
            conditions.append(f"STARTSWITH(c.namespace, @namespace)")
            conditions.append(f"c.partition_key = @partition_key")
            query_params.append({"name": "@namespace", "value": namespace_str})
            query_params.append({"name": "@partition_key", "value": partition_key})
        
        # Apply additional filters
        if filter:
            for key, value in filter.items():
                if isinstance(value, dict):
                    for op_name, val in value.items():
                        condition, params = self._get_filter_condition(key, op_name, val)
                        conditions.append(condition)
                        query_params.extend([{"name": "@value", "value": p} for p in params])
                else:
                    conditions.append(f"c.value['{key}'] = @{key}")
                    query_params.append({"name": f"@{key}", "value": value})
        
        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))
        
        # Vector search
        if query and self.index_config and self.embeddings:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            query_parts[0] = (
                f"SELECT TOP {limit} c.id, c.partition_key, c.namespace, c.key, c['value'], "
                f"c.created_at, c.updated_at, c.expires_at, c.ttl_seconds, "
                f"VectorDistance(c.embedding, @query_embedding, false) AS score FROM c"
            )
            query_params.append({
                "name": "@query_embedding",
                "value": query_embedding
            })
            query_parts.append(f"ORDER BY VectorDistance(c.embedding, @query_embedding, false)")
        else:
            query_parts.append("ORDER BY c.updated_at DESC")
            query_parts.append(f"OFFSET {offset} LIMIT {limit}")
        
        
        query_str = " ".join(query_parts)
        items = list(self.container.query_items(
            query=query_str,
            parameters=query_params
        ))
        
        # Refresh TTL if requested
        if refresh_ttl:
            for item in items:
                if item.get("ttl_seconds"):
                    item["expires_at"] = self._calculate_ttl_expiry(
                        item["ttl_seconds"] / 60
                    )
                    item["updated_at"] = self._get_current_timestamp()
                    self.container.upsert_item(self._serialize_item(item))
        
        return [self._row_to_search_item(item) for item in items]
    
    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item from the store.
        
        Args:
            namespace: Namespace tuple
            key: Item key
        """
        namespace_str = self._namespace_to_text(namespace)
        partition_key = self._get_partition_key(namespace)
        doc_id = f"{namespace_str}__{key}"
        
        try:
            self.container.delete_item(
                item=doc_id,
                partition_key=partition_key
            )
        except cosmos_exceptions.CosmosResourceNotFoundError:
            pass
    
    def list_namespaces(
        self,
        *,
        prefix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List distinct namespaces in the store.
        
        Args:
            prefix: Namespace prefix filter
            max_depth: Maximum depth to return
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            List of namespace tuples
        """
        query_parts = ["SELECT DISTINCT c.namespace FROM c"]
        query_params: list[dict[str, Any]] = []
        
        if prefix:
            prefix_str = self._namespace_to_text(prefix)
            query_parts.append("WHERE STARTSWITH(c.namespace, @prefix)")
            query_params.append({"name": "@prefix", "value": prefix_str})
        
        query_parts.append(f"OFFSET {offset} LIMIT {limit}")
        query_str = " ".join(query_parts)
        
        items = list(self.container.query_items(
            query=query_str,
            parameters=query_params
        ))
        
        namespaces = []
        for item in items:
            ns = self._text_to_namespace(item["namespace"])
            
            # Apply max_depth if specified
            if max_depth is not None and len(ns) > max_depth:
                ns = ns[:max_depth]
            
            if ns not in namespaces:
                namespaces.append(ns)
        
        return namespaces
    
    def batch(self, ops: Iterable[GetOp | SearchOp | PutOp | ListNamespacesOp]) -> list[Item | list[Item] | list[SearchItem] | list[tuple[str, ...]] | None]:
        """Execute a batch of operations.
        
        Args:
            ops: Iterable of operations (GetOp, PutOp, SearchOp, etc.)
            
        Returns:
            List of results corresponding to each operation
        """
        results: list[Any] = []
        
        for op in ops:
            op_type = type(op).__name__
            
            if op_type == "GetOp":
                result = self.get(
                    op.namespace,
                    op.key,
                    refresh_ttl=getattr(op, "refresh_ttl", False)
                )
                results.append(result)
            
            elif op_type == "PutOp":
                self.put(
                    op.namespace,
                    op.key,
                    op.value,
                    index=getattr(op, "index", None),
                    ttl=getattr(op, "ttl", None)
                )
                results.append(None)
            
            elif op_type == "SearchOp":
                search_results = self.search(
                    op.namespace_prefix,
                    query=getattr(op, "query", None),
                    filter=getattr(op, "filter", None),
                    limit=getattr(op, "limit", 10),
                    offset=getattr(op, "offset", 0),
                    refresh_ttl=getattr(op, "refresh_ttl", False)
                )
                results.append(search_results)
            
            elif op_type == "ListNamespacesOp":
                ns_results = self.list_namespaces(
                    prefix=getattr(op, "prefix", None),
                    max_depth=getattr(op, "max_depth", None),
                    limit=getattr(op, "limit", 100),
                    offset=getattr(op, "offset", 0)
                )
                results.append(ns_results)
            
            else:
                raise ValueError(f"Unsupported operation type: {op_type}")
        
        return results
    
    def sweep_ttl(self) -> int:
        """Delete expired items based on TTL.
        
        Returns:
            Number of deleted items
        """
        now = self._get_current_timestamp()
        
        # Query expired items
        query = """
        SELECT c.id, c.partition_key
        FROM c
        WHERE c.expires_at != null AND c.expires_at < @now
        """
        
        items = list(self.container.query_items(
            query=query,
            parameters=[{"name": "@now", "value": now}]
        ))
        
        # Delete expired items
        deleted_count = 0
        for item in items:
            try:
                self.container.delete_item(
                    item=item["id"],
                    partition_key=item["partition_key"]
                )
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete expired item {item['id']}: {e}")
        
        return deleted_count
    
    def start_ttl_sweeper(
        self, sweep_interval_minutes: Optional[int] = None
    ) -> concurrent.futures.Future[None]:
        """Start background thread to periodically delete expired items.
        
        Args:
            sweep_interval_minutes: Interval between sweeps (default: 5)
            
        Returns:
            Future that can be waited on or cancelled
        """
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future
        
        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            logger.info("TTL sweeper thread is already running")
            future = concurrent.futures.Future()
            future.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return future
        
        self._ttl_stop_event.clear()
        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes", 5)
        )
        
        logger.info(f"Starting store TTL sweeper with interval {interval} minutes")
        future = concurrent.futures.Future()
        
        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break
                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info(f"Store swept {expired_items} expired items")
                    except Exception as exc:
                        logger.exception("Store TTL sweep iteration failed", exc_info=exc)
                future.set_result(None)
            except Exception as exc:
                future.set_exception(exc)
        
        thread = threading.Thread(target=_sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        thread.start()
        
        future.add_done_callback(
            lambda f: self._ttl_stop_event.set() if f.cancelled() else None
        )
        
        return future
    
    def stop_ttl_sweeper(self, timeout: Optional[float] = None) -> bool:
        """Stop the TTL sweeper thread.
        
        Args:
            timeout: Maximum time to wait for thread to stop (seconds)
            
        Returns:
            True if successfully stopped, False if timeout reached
        """
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True
        
        logger.info("Stopping TTL sweeper thread")
        self._ttl_stop_event.set()
        self._ttl_sweeper_thread.join(timeout)
        
        success = not self._ttl_sweeper_thread.is_alive()
        if success:
            self._ttl_sweeper_thread = None
            logger.info("TTL sweeper thread stopped")
        else:
            logger.warning("Timed out waiting for TTL sweeper thread to stop")
        
        return success
    
    def __del__(self) -> None:
        """Cleanup when object is garbage collected."""
        if hasattr(self, "_ttl_stop_event") and hasattr(self, "_ttl_sweeper_thread"):
            self.stop_ttl_sweeper(timeout=0.1)


