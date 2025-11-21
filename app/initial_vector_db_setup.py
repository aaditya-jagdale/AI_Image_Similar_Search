from app.core.config import CHROMA_DB_NAME, CHROMA_DIR, SENTENCE_TRANSFORMER_MODEL, ONNX_MODEL_PATH 
from app.services.image_vectorization_service import ImageVectorizerONNX
from app.models.add_document_models import AddDocumentModels
from typing import Dict

def add_documents_to_chroma():
    documents = []
    with open('data/formatted_products_rows.csv', mode='r', encoding='utf-8') as file:
        import csv
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            documents.append(
                AddDocumentModels(
                    image_uri=row["image"],
                    metadata=eval(row["metadata"]),
                    id=row["id"]
                )
            )
        print(f"Loaded {len(documents)} documents from CSV.")

    chroma_service.add_images_from_urls(
        image_data=documents
    )

def query_images_from_chroma(image: str, n_results: int =5) -> list[Dict]:
    results = chroma_service.query_by_image(
        image_path=image,
        n_results=n_results
    )

    return results

if __name__ == "__main__":
    add_documents_to_chroma()