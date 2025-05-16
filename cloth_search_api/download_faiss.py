import os
from supabase_service import SupabaseService

def download_faiss_files():
    """
    Downloads FAISS index files from Supabase storage.
    Returns a tuple of (success, message) where success is a boolean.
    """
    try:
        # Initialize Supabase service
        supabase = SupabaseService()
        
        # Define file paths
        local_index_path = os.path.join("faiss/search.faiss")
        local_metadata_path = os.path.join("metadata/search_metadata.json")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(local_index_path), exist_ok=True)
        os.makedirs(os.path.dirname(local_metadata_path), exist_ok=True)
        
        # Download FAISS index file
        print("📥 Downloading FAISS index file...")
        index_data = supabase.supabase.storage.from_("products").download("faiss/search.faiss")
        with open(local_index_path, "wb") as f:
            f.write(index_data)
        print("✅ FAISS index file downloaded successfully at", local_index_path)
        
        # Download metadata file
        print("📥 Downloading metadata file...")
        metadata_data = supabase.supabase.storage.from_("products").download("faiss/search_metadata.json")
        with open(local_metadata_path, "wb") as f:
            f.write(metadata_data)
        print("✅ Metadata file downloaded successfully at", local_metadata_path)
        
        return True, "FAISS index files downloaded successfully"
        
    except Exception as e:
        error_message = f"Error downloading FAISS index files: {str(e)}"
        print(f"❌ {error_message}")
        return False, error_message 