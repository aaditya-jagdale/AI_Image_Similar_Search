from agno.vectordb.chroma import ChromaDb
from app.core.config import CHROMA_DB_NAME, CHROMA_DIR, GEMINI_API_KEY
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.embedder.google import GeminiEmbedder

geminiEmbedder = GeminiEmbedder(
    api_key=GEMINI_API_KEY,
)

chromadb = ChromaDb(
    name="textile_designs",
    persistent_client=True,
    description="Image embeddings stored of all the textile designs",
    collection=CHROMA_DB_NAME,
    path=CHROMA_DIR,
    embedder=geminiEmbedder
)

knowledge = Knowledge(
    name="textile_design_knowledge_base",
    description="Knowledge base for textile design image embeddings",
    vector_db=chromadb,
)

class VectorDatabaseService:
    def __init__(self):
        self.db = chromadb
        self.knowledge = knowledge

    def add_embedding(self, embedding: list[list[float]], metadata: dict):
        self.knowledge.add_contents(
            embedding=embedding,
            metadata=metadata,
            upsert=True,
            skip_if_exists=True,
        )
    
    def fetch_content(self, query_embedding: list[float], topk: int = 5) -> list[dict]:
        results = self.knowledge.get_content(
            query_embedding=query_embedding,
            sort_by='similarity',
            limit=topk,
        )
        return results