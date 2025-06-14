from ..text_search import metadata_search
from ..models.request_models import MetadataSearchRequest
import chromadb
import os

async def perform_metadata_search(design_no, width, stock, GSM, source_pdf, user_id, top_k):
    search_request = MetadataSearchRequest(
        design_no=design_no,
        width=width,
        stock=stock,
        GSM=GSM,
        source_pdf=source_pdf
    )
    chroma_dir = os.path.join(os.getcwd(), 'chroma')
    client = chromadb.PersistentClient(path=chroma_dir)
    return await metadata_search(search_request, client, user_id, top_k)
