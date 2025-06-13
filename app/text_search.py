import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
import os
from typing import List, Dict
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)
load_dotenv()

class MetadataSearchRequest(BaseModel):
    source_pdf: str
    design_no: str
    width: str
    stock: str
    GSM: str

# Initialize the embedding function
embedding_function = SentenceTransformerEmbeddingFunction()
async def metadata_search(query: MetadataSearchRequest, client: chromadb.PersistentClient, user_id: str, top_k: int = 10) -> List[Dict]:
    
    collection = client.get_collection(name=user_id)
    
    # Build query dict only with non-empty fields
    query_dict = {}
    for field, value in query.model_dump().items():
        if value and value.strip():
            query_dict[field] = value
            
    if not query_dict:
        return []
        
    # Convert dict to string for query
    query_str = " AND ".join([f"{k}:{v}" for k,v in query_dict.items()])
    
    results = collection.query(
        query_texts=[query_str],
        n_results=top_k
    )
    
    return results.get("metadatas", [])

if __name__ == "__main__":
    # Test the search functionality
    client = chromadb.PersistentClient(path="./chroma")
    import asyncio
    
    async def test_search():
        results = await text_search(
            query="I want a design with width 58 and gsm 200",
            client=client,
            user_id="test_user"
        )
        print("Search Results:", results)
    
    asyncio.run(test_search())