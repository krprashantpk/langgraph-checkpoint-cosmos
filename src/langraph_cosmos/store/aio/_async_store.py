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

if TYPE_CHECKING:
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
        TTLConfig,
    )

from ..base import BaseCosmosStore, CosmosIndexConfig

logger = logging.getLogger(__name__)


class AsyncCosmosStore(BaseCosmosStore[AsyncCosmosClient]):
    """Async Cosmos DB-backed store with optional vector search.
    
    Similar to CosmosStore but with async/await support for all operations.
    
    Example:
        ```python
        from azure.cosmos.aio import CosmosClient
        from store.cosmos.base import AsyncCosmosStore
        
        async def main():
            client = CosmosClient(
                url="https://your-account.documents.azure.com:443/",
                credential="your-key"
            )
            
            store = AsyncCosmosStore(
                client=client,
                database_name="my_database",
                container_name="my_container"
            )
            
            await store.setup()
            
            # Store documents
            await store.aput(("docs",), "doc1", {"text": "Python tutorial"})
            
            # Get document
            doc = await store.aget(("docs",), "doc1")
            
            # Search
            results = await store.asearch(("docs",), limit=10)
            
            await client.close()
        
        import asyncio
        asyncio.run(main())
        ```
    """
    
    __slots__ = ("database", "container", "_ttl_sweeper_task")
    
    supports_ttl: bool = True
    
    def __init__(
        self,
        client: AsyncCosmosClient,
        database_name: str,
        container_name: str,
        *,
        deserializer: Callable[[bytes | str], dict[str, Any]] | None = None,
        index: CosmosIndexConfig | None = None,
        ttl: Optional[dict[str, Any]] = None,
    ):
        """Initialize async Cosmos DB store."""
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
        self._ttl_sweeper_task: Optional[asyncio.Task] = None
    
    @classmethod
    @asynccontextmanager
    async def from_connection_string(
        cls,
        connection_string: str,
        database_name: str,
        container_name: str,
        *,
        index: CosmosIndexConfig | None = None,
        ttl: Optional[dict[str, Any]] = None,
    ):
        """Create async CosmosStore from connection string.
        
        Args:
            connection_string: Cosmos DB connection string
            database_name: Name of the database
            container_name: Name of the container
            index: Vector index configuration
            ttl: TTL configuration
            
        Yields:
            AsyncCosmosStore instance
        """
        client = AsyncCosmosClient.from_connection_string(connection_string)
        try:
            yield cls(
                client=client,
                database_name=database_name,
                container_name=container_name,
                index=index,
                ttl=ttl,
            )
        finally:
            await client.close()
    
    async def setup(self) -> None:
        """Set up the Cosmos DB container asynchronously."""
        # Create database if it doesn't exist
        try:
            await self.client.create_database(self.database_name)
            logger.info(f"Created database: {self.database_name}")
        except cosmos_exceptions.CosmosResourceExistsError:
            logger.debug(f"Database already exists: {self.database_name}")
        
        # Similar setup logic as sync version
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": '/"_etag"/?'}],
        }
        
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
            
            if "vectorIndexes" not in indexing_policy:
                indexing_policy["vectorIndexes"] = []
            
            indexing_policy["vectorIndexes"].append({
                "path": "/embedding",
                "type": vector_index_type,
            })
        
        try:
            # Use proper parameter names for async SDK
            await self.database.create_container(
                id=self.container_name,
                partition_key=PartitionKey(path="/partition_key"),
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy
            )
            logger.info(f"Created container: {self.container_name}")
        except cosmos_exceptions.CosmosResourceExistsError:
            logger.debug(f"Container already exists: {self.container_name}")
    
    async def aget(
        self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool = False
    ) -> Optional[Item]:
        """Async get item from store."""
        namespace_str = self._namespace_to_text(namespace)
        partition_key = self._get_partition_key(namespace)
        doc_id = f"{namespace_str}__{key}"
        
        try:
            item = await self.container.read_item(
                item=doc_id,
                partition_key=partition_key
            )
            
            if refresh_ttl and item.get("ttl_seconds"):
                item["expires_at"] = self._calculate_ttl_expiry(
                    item["ttl_seconds"] / 60
                )
                item["updated_at"] = self._get_current_timestamp()
                await self.container.upsert_item(self._serialize_item(item))
            
            return self._row_to_item(item)
            
        except cosmos_exceptions.CosmosResourceNotFoundError:
            return None
    
    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        *,
        index: Optional[bool | list[str]] = None,
        ttl: Optional[float] = None,
    ) -> None:
        """Async store item in the store."""
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
        
        if ttl is not None:
            ttl_seconds = int(ttl * 60)
            item["ttl_seconds"] = ttl_seconds
            item["expires_at"] = self._calculate_ttl_expiry(ttl)
        
        # Handle embeddings (similar to sync version)
        if self.index_config and self.embeddings and index is not False:
            if index is None:
                fields = self.index_config.get("fields", [])
            else:
                fields = index if isinstance(index, list) else self.index_config.get("fields", [])
            
            texts_to_embed = []
            for field in fields:
                text = self._extract_text_from_path(value, field)
                if text:
                    texts_to_embed.append(text)
            
            if texts_to_embed:
                combined_text = "\n\n".join(texts_to_embed)
                # Async embed if available
                if hasattr(self.embeddings, "aembed_documents"):
                    embeddings = await self.embeddings.aembed_documents([combined_text])
                else:
                    embeddings = self.embeddings.embed_documents([combined_text])
                
                item["embedding"] = embeddings[0]
        
        await self.container.upsert_item(self._serialize_item(item))
    
    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        *,
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool = False,
    ) -> list[SearchItem]:
        """Async search for items."""
        namespace_str = self._namespace_to_text(namespace_prefix)
        
        query_parts = ["SELECT * FROM c"]
        query_params: list[dict[str, Any]] = []
        conditions = []
        
        if namespace_prefix:
            partition_key = self._get_partition_key(namespace_prefix)
            conditions.append(f"STARTSWITH(c.namespace, @namespace)")
            conditions.append(f"c.partition_key = @partition_key")
            query_params.append({"name": "@namespace", "value": namespace_str})
            query_params.append({"name": "@partition_key", "value": partition_key})
        
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
        
        if query and self.index_config and self.embeddings:
            # Async embed if available
            if hasattr(self.embeddings, "aembed_query"):
                query_embedding = await self.embeddings.aembed_query(query)
            else:
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
        items = []
        async for item in self.container.query_items(
            query=query_str,
            parameters=query_params
        ):
            items.append(item)
        if refresh_ttl:
            for item in items:
                if item.get("ttl_seconds"):
                    item["expires_at"] = self._calculate_ttl_expiry(
                        item["ttl_seconds"] / 60
                    )
                    item["updated_at"] = self._get_current_timestamp()
                    await self.container.upsert_item(self._serialize_item(item))
        
        return [self._row_to_search_item(item) for item in items]
    
    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Async delete item from store."""
        namespace_str = self._namespace_to_text(namespace)
        partition_key = self._get_partition_key(namespace)
        doc_id = f"{namespace_str}__{key}"
        
        try:
            await self.container.delete_item(
                item=doc_id,
                partition_key=partition_key
            )
        except cosmos_exceptions.CosmosResourceNotFoundError:
            pass
    
    async def alist_namespaces(
        self,
        *,
        prefix: Optional[tuple[str, ...]] = None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """Async list namespaces."""
        query_parts = ["SELECT DISTINCT c.namespace FROM c"]
        query_params: list[dict[str, Any]] = []
        
        if prefix:
            prefix_str = self._namespace_to_text(prefix)
            query_parts.append("WHERE STARTSWITH(c.namespace, @prefix)")
            query_params.append({"name": "@prefix", "value": prefix_str})
        
        query_parts.append(f"OFFSET {offset} LIMIT {limit}")
        query_str = " ".join(query_parts)
        
        items = []
        async for item in self.container.query_items(
            query=query_str,
            parameters=query_params
        ):
            items.append(item)
        
        namespaces = []
        for item in items:
            ns = self._text_to_namespace(item["namespace"])
            
            if max_depth is not None and len(ns) > max_depth:
                ns = ns[:max_depth]
            
            if ns not in namespaces:
                namespaces.append(ns)
        
        return namespaces
    
    async def abatch(self, ops: Iterable[Any]) -> list[Item | list[Item] | list[SearchItem] | list[tuple[str, ...]] | None]:
        """Async execute batch of operations."""
        results: list[Any] = []
        
        for op in ops:
            op_type = type(op).__name__
            
            if op_type == "GetOp":
                
                result = await self.aget(
                    op.namespace,
                    op.key,
                    refresh_ttl=getattr(op, "refresh_ttl", False)
                )
                results.append(result)
            
            elif op_type == "PutOp":
                await self.aput(
                    op.namespace,
                    op.key,
                    op.value,
                    index=getattr(op, "index", None),
                    ttl=getattr(op, "ttl", None)
                )
                results.append(None)
            
            elif op_type == "SearchOp":
                search_results = await self.asearch(
                    op.namespace_prefix,
                    query=getattr(op, "query", None),
                    filter=getattr(op, "filter", None),
                    limit=getattr(op, "limit", 10),
                    offset=getattr(op, "offset", 0),
                    refresh_ttl=getattr(op, "refresh_ttl", False)
                )
                results.append(search_results)
            
            elif op_type == "ListNamespacesOp":
                ns_results = await self.alist_namespaces(
                    prefix=getattr(op, "prefix", None),
                    max_depth=getattr(op, "max_depth", None),
                    limit=getattr(op, "limit", 100),
                    offset=getattr(op, "offset", 0)
                )
                results.append(ns_results)
            
            else:
                raise ValueError(f"Unsupported operation type: {op_type}")
        
        return results
    
    async def asweep_ttl(self) -> int:
        """Async delete expired items."""
        now = self._get_current_timestamp()
        
        query = """
        SELECT c.id, c.partition_key
        FROM c
        WHERE c.expires_at != null AND c.expires_at < @now
        """
        
        items = []
        async for item in self.container.query_items(
            query=query,
            parameters=[{"name": "@now", "value": now}]
        ):
            items.append(item)
        
        deleted_count = 0
        for item in items:
            try:
                await self.container.delete_item(
                    item=item["id"],
                    partition_key=item["partition_key"]
                )
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete expired item {item['id']}: {e}")
        
        return deleted_count
    
    async def astart_ttl_sweeper(
        self, sweep_interval_minutes: Optional[int] = None
    ) -> None:
        """Start background task to periodically delete expired items."""
        if not self.ttl_config:
            return
        
        if self._ttl_sweeper_task and not self._ttl_sweeper_task.done():
            logger.info("TTL sweeper task is already running")
            return
        
        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes", 5)
        )
        
        logger.info(f"Starting async store TTL sweeper with interval {interval} minutes")
        
        async def _sweep_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(interval * 60)
                    expired_items = await self.asweep_ttl()
                    if expired_items > 0:
                        logger.info(f"Store swept {expired_items} expired items")
                except asyncio.CancelledError:
                    logger.info("TTL sweeper task cancelled")
                    break
                except Exception as exc:
                    logger.exception("Store TTL sweep iteration failed", exc_info=exc)
        
        self._ttl_sweeper_task = asyncio.create_task(_sweep_loop())
    
    async def astop_ttl_sweeper(self) -> None:
        """Stop the TTL sweeper task."""
        if self._ttl_sweeper_task and not self._ttl_sweeper_task.done():
            logger.info("Stopping TTL sweeper task")
            self._ttl_sweeper_task.cancel()
            try:
                await self._ttl_sweeper_task
            except asyncio.CancelledError:
                pass
            self._ttl_sweeper_task = None
            logger.info("TTL sweeper task stopped")

