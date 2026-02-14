import os
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding

load_dotenv()

# Initialize Cosmos client
client = CosmosClient(os.environ["COSMOS_ENDPOINT"], os.environ["COSMOS_KEY"])
print("✅ Using Cosmos DB connection key for ingestion")
db = client.create_database_if_not_exists(id=os.environ["COSMOS_DB"])
container = db.create_container_if_not_exists(
    id=os.environ["COSMOS_CONTAINER"],
    partition_key=PartitionKey(path=os.environ["COSMOS_PARTITION_KEY"]),
    indexing_policy={
        "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}]
    },
    vector_embedding_policy={
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "dimensions": 1536,
                "distanceFunction": "cosine"
            }
        ]
    }
)

# Initialize Semantic Kernel for embeddings
def create_embedding_kernel():
    kernel = Kernel()
    kernel.add_service(
        AzureTextEmbedding(
            deployment_name=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"]
        )
    )
    return kernel

def embed_texts(texts):
    """
    Generate embeddings using Semantic Kernel.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each is a list of floats)
    """
    import asyncio

    async def _generate_embeddings():
        # Create kernel with embedding service
        kernel = create_embedding_kernel()

        # Get embedding service from kernel
        embed_service = kernel.get_service(type=AzureTextEmbedding)

        embeddings = []
        for text in texts:
            # Generate embedding for each text
            result = await embed_service.generate_embeddings([text])
            # Convert to list format for JSON serialization
            embedding_vector = list(result[0])
            embeddings.append(embedding_vector)

        return embeddings

    # Run async function
    return asyncio.run(_generate_embeddings())

def upsert_snippet(doc_id, text, pk="cards"):
    """
    Upsert a document with its embedding into Cosmos DB.

    Args:
        doc_id: Unique document identifier
        text: The text content to store and embed
        pk: Partition key value (default: "cards")
    """
    try:
        # Generate embedding for the text
        embeddings = embed_texts([text])
        embedding_vector = embeddings[0] if embeddings else []

        # Create document dict with id, pk, text, and embedding fields
        document = {
            "id": doc_id,
            "pk": pk,
            "text": text,
            "embedding": embedding_vector
        }

        # Upsert document into Cosmos DB
        container.upsert_item(body=document)

        print(f"✅ {doc_id} upserted with Semantic Kernel embeddings.")
    except Exception as e:
        print(f"❌ Failed to upsert {doc_id}: {e}")

if __name__ == "__main__":
    upsert_snippet("bankgold-1", "BankGold: 4x points on dining worldwide; no FX fees.")
    upsert_snippet("lounge-1", "Lounge access in CDG Terminal 2 with BankGold Premium.")
    upsert_snippet("bankgold-2", "BankGold Premium: Priority boarding, lounge access, concierge service.")
    upsert_snippet("dining-1", "4x points on dining worldwide with BankGold card.")
    print("✅ All snippets upserted with Semantic Kernel embeddings.")
