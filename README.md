# LangGraph Cosmos DB

[![PyPI](https://img.shields.io/pypi/v/langraph-cosmos.svg)](https://pypi.org/project/langraph-cosmos/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Azure Cosmos DB implementation of LangGraph checkpoint saver and store for building stateful, multi-actor applications with LLMs.

## Overview

This library provides a production-ready integration between [LangGraph](https://github.com/langchain-ai/langgraph) and [Azure Cosmos DB](https://azure.microsoft.com/en-us/products/cosmos-db/), enabling:

- **Durable checkpoint storage** for LangGraph agents (coming soon)
- **Persistent memory store** with support for both sync and async operations
- **Vector search** capabilities using Cosmos DB's native vector indexing
- **Automatic TTL** (Time-To-Live) for managing data lifecycle
- **Global distribution** with Azure Cosmos DB's multi-region replication

## Why Cosmos DB for LangGraph?

Azure Cosmos DB is an ideal backend for LangGraph applications:

- **🌍 Global Distribution**: Deploy your AI agents globally with multi-region writes and reads
- **⚡ Low Latency**: Single-digit millisecond response times for state access
- **🔍 Vector Search**: Native support for vector embeddings with DiskANN, quantized flat, and flat indexes
- **📈 Elastic Scale**: Automatic scaling from zero to planet-scale throughput
- **💾 Flexible Schema**: Store complex agent state with JSON documents
- **🔒 Enterprise-Ready**: Built-in security, compliance, and 99.999% SLA

## Installation

```bash
pip install langraph-cosmos
```

### Optional Dependencies

For vector search functionality:

```bash
pip install langchain-openai  # or your preferred embeddings provider
```

## Quick Start

### Synchronous Store

```python
from azure.cosmos import CosmosClient
from langraph_cosmos.store import CosmosStore

# Create Cosmos DB client
client = CosmosClient(
    url="https://your-account.documents.azure.com:443/",
    credential="your-primary-key"
)

# Initialize store
store = CosmosStore(
    client=client,
    database_name="langgraph_db",
    container_name="memory_store"
)

# Setup container (creates if doesn't exist)
store.setup()

# Store data
store.put(
    namespace=("users", "user123"),
    key="preferences",
    value={"theme": "dark", "language": "en"}
)

# Retrieve data
item = store.get(
    namespace=("users", "user123"),
    key="preferences"
)
print(item.value)  # {'theme': 'dark', 'language': 'en'}

# List namespaces
namespaces = list(store.list_namespaces(prefix=("users",)))
print(namespaces)  # [('users', 'user123')]

# Search within namespace
results = list(store.search(
    namespace_prefix=("users",),
    limit=10
))
```

### Asynchronous Store

```python
import asyncio
from azure.cosmos.aio import CosmosClient
from langraph_cosmos.store import aio

async def main():
    # Create async client
    client = CosmosClient(
        url="https://your-account.documents.azure.com:443/",
        credential="your-primary-key"
    )

    # Initialize async store
    store = aio.CosmosStore(
        client=client,
        database_name="langgraph_db",
        container_name="memory_store"
    )

    await store.setup()

    # Store data asynchronously
    await store.aput(
        namespace=("agents", "agent_1"),
        key="state",
        value={"step": 5, "context": "processing"}
    )

    # Retrieve data
    item = await store.aget(
        namespace=("agents", "agent_1"),
        key="state"
    )
    print(item.value)

    # Batch operations
    items = [
        ("agents", "agent_1", "config", {"temperature": 0.7}),
        ("agents", "agent_2", "config", {"temperature": 0.9}),
    ]

    await store.abatch([
        {"namespace": ns, "key": k, "value": v}
        for ns, _, k, v in [(tuple(i[0].split(",")), i[1], i[2], i[3]) for i in items]
    ])

    await client.close()

asyncio.run(main())
```

### Connection String

Both sync and async stores support connection strings:

```python
from langraph_cosmos.store import CosmosStore

with CosmosStore.from_connection_string(
    connection_string="AccountEndpoint=https://...;AccountKey=...;",
    database_name="langgraph_db",
    container_name="memory_store"
) as store:
    store.setup()
    store.put(("test",), "key1", {"data": "value"})
```

## Advanced Features

### Vector Search

Enable semantic search capabilities with vector embeddings:

```python
from azure.cosmos import CosmosClient
from langchain_openai import OpenAIEmbeddings
from langraph_cosmos.store import CosmosStore, CosmosIndexConfig

# Configure vector indexing
index_config = CosmosIndexConfig(
    embed=OpenAIEmbeddings(model="text-embedding-3-small"),
    fields=["text", "content"],  # Fields to embed
    dims=1536,  # Embedding dimensions
    vector_index_type="diskANN",  # Options: flat, diskANN, quantizedFlat
    distance_type="cosine"  # Options: cosine, euclidean, dotproduct
)

store = CosmosStore(
    client=client,
    database_name="langgraph_db",
    container_name="vector_store",
    index=index_config
)

store.setup()

# Store documents with automatic embedding
store.put(
    namespace=("documents",),
    key="doc1",
    value={
        "text": "LangGraph is a framework for building stateful agents",
        "metadata": {"source": "docs"}
    }
)

store.put(
    namespace=("documents",),
    key="doc2",
    value={
        "text": "Azure Cosmos DB provides global distribution and low latency",
        "metadata": {"source": "azure"}
    }
)

# Semantic search
results = list(store.search(
    namespace_prefix=("documents",),
    query="agent frameworks",
    limit=5
))

for item in results:
    print(f"Score: {item.score}, Text: {item.value['text']}")
```

### Quantized Vector Search

For reduced memory footprint:

```python
index_config = CosmosIndexConfig(
    embed=OpenAIEmbeddings(),
    fields=["text"],
    dims=1536,
    vector_index_type="quantizedFlat",
    distance_type="cosine",
    quantization_byte_size=2  # 1, 2, or 4 bytes
)
```

### TTL (Time-To-Live)

Automatically expire old data:

```python
from langraph_cosmos.store import CosmosStore

store = CosmosStore(
    client=client,
    database_name="langgraph_db",
    container_name="ephemeral_store",
    ttl={"default_ttl": 60}  # 60 minutes
)

store.setup()

# Start TTL sweeper thread (required for TTL to work)
store.start_ttl_sweeper()

# Store with custom TTL
store.put(
    namespace=("sessions",),
    key="session_123",
    value={"user": "alice", "active": True}
    # Will expire after 60 minutes
)

# Later, stop the sweeper
store.stop_ttl_sweeper()
```

For async:

```python
# Start TTL sweeper task
await store.start_ttl_sweeper()

# Later, stop it
await store.stop_ttl_sweeper()
```

### Filtering

Search with metadata filters:

```python
# Store items with filterable metadata
store.put(
    namespace=("products",),
    key="prod1",
    value={"name": "Widget", "price": 19.99, "category": "tools"}
)

store.put(
    namespace=("products",),
    key="prod2",
    value={"name": "Gadget", "price": 29.99, "category": "electronics"}
)

# Search with filters
results = list(store.search(
    namespace_prefix=("products",),
    filter={"category": {"$eq": "tools"}}
))
```

Supported filter operators:

- `$eq`: Equal to
- `$ne`: Not equal to
- `$gt`: Greater than
- `$gte`: Greater than or equal
- `$lt`: Less than
- `$lte`: Less than or equal

## Resources

- **LangGraph**: [Documentation](https://docs.langchain.com/oss/python/langgraph/overview) | [GitHub](https://github.com/langchain-ai/langgraph)
- **Azure Cosmos DB**: [Documentation](https://learn.microsoft.com/azure/cosmos-db/) | [Python SDK](https://github.com/Azure/azure-sdk-for-python/tree/main/sdk/cosmos/azure-cosmos)
- **Vector Search**: [Cosmos DB Vector Search Guide](https://learn.microsoft.com/azure/cosmos-db/vector-search)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built on top of:

- [LangGraph](https://github.com/langchain-ai/langgraph) - The LangChain orchestration framework
- [Azure Cosmos DB](https://azure.microsoft.com/products/cosmos-db/) - Microsoft's globally distributed database
- [LangChain](https://github.com/langchain-ai/langchain) - For embeddings integration

## Support

For issues and questions:

- **GitHub Issues**: [Report bugs or request features](https://github.com/yourusername/langraph-checkpoint-cosmos/issues)
- **LangChain Forum**: [Community discussions](https://forum.langchain.com/)
- **Azure Support**: [Azure Cosmos DB support](https://azure.microsoft.com/support/)

---

Built with ❤️ for the LangGraph community
