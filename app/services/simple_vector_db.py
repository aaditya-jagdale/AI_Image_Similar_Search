import chromadb
from app.core.config import CHROMA_DB_NAME, CHROMA_DIR
import numpy as np
from typing import List, Dict, Any

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
        print(f"ChromaDB service initialized. Collection '{CHROMA_DB_NAME}' ready.")

    def add_image(self, embedding: np.ndarray, metadata: Dict[str, Any], id: str):
        """
        Adds a single image embedding, metadata, and ID to the collection.
        
        Args:
            embedding: The numpy array (1, D) of the image embedding.
            metadata: A dictionary of metadata for the image.
            id: A unique string ID for the image.
        """
        # ChromaDB expects a list of embeddings and a list of metadatas/ids.
        # We convert the (1, D) numpy array to a simple list.
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
        """
        Queries the collection for the most similar image embeddings.
        
        Args:
            query_embedding: The numpy array (1, D) of the query image embedding.
            top_k: The number of similar results to return.
            
        Returns:
            A list of dictionaries containing search results.
        """
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