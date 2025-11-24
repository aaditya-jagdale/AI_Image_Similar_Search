import chromadb
from app.core.config import CHROMA_DB_NAME, CHROMA_DIR
import numpy as np
from agno.knowledge import Knowledge
from agno.vectordb.upstashdb import UpstashVectorDb
from typing import List, Dict, Any
import os

vector_db = UpstashVectorDb(
    url=os.getenv("UPSTASH_VECTOR_REST_URL"),
    token=os.getenv("UPSTASH_VECTOR_REST_TOKEN"),
)

# Initialize Upstash DB
knowledge = Knowledge(
    name="Basic SDK Knowledge Base",
    description="Agno 2.0 Knowledge Implementation with Upstash Vector DB",
    vector_db=vector_db,
)

class SimpleVectorDB:
    def __init__(self):
        """
        Initializes a persistent ChromaDB client and gets or creates the collection.
        """
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_DB_NAME
        )
        print(f"ChromaDB service initialized. Collection ready. db_name: '{CHROMA_DB_NAME}' db_path: '{CHROMA_DIR}'")

    def add_image(self, embedding: np.ndarray, metadata: Dict[str, Any], id: str):
        if embedding.ndim > 1:
            embedding_list = embedding.tolist()[0]
        else:
            embedding_list = embedding.tolist()

        try:
            self.collection.add(
                embeddings=[embedding_list],
                metadatas=[metadata],
                ids=[id]
            )
            print(f"Successfully added image with ID: {id}")
        except chromadb.errors.IDAlreadyExistsError:
            print(f"Image with ID: {id} already exists. Skipping.")
        except Exception as e:
            print(f"Error adding image {id}: {e}")

    def query_image(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if query_embedding.ndim > 1:
            query_list = query_embedding.tolist()[0]
        else:
            query_list = query_embedding.tolist()

        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        return results


class UpstashVectorDataService:
    def __init__(self):
        pass

    def add_item(self, image: str, metadata: Dict[str, Any], id: str):
        knowledge.add_content(
            url=image,
            metadata=metadata,
            id=id,
        )
