from ..chroma_provider import search_images

def perform_search(user_id: str, image_url: str, top_k: int):
    return search_images(user_id=user_id, image_path_or_url=image_url, top_k=top_k)
