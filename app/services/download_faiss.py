import os
from .supabase_service import SupabaseService
from dotenv import load_dotenv
import json

load_dotenv()

# ------------------------- Download FAISS Files -------------------------
def download_faiss_files():
    """
    Downloads FAISS index files from Supabase storage.
    Returns a tuple of (success, message) where success is a boolean.
    """
    try:
        # Initialize Supabase service
        supabase = SupabaseService()
        
        # Define local paths
        faiss_dir = "faiss_indices"
        os.makedirs(faiss_dir, exist_ok=True)
        
        # Define file paths
        local_index_path = os.path.join(faiss_dir, "search.faiss")
        local_metadata_path = os.path.join(faiss_dir, "search_metadata.json")
        
        # Download FAISS index file
        print("📥 Downloading FAISS index file...")
        index_data = supabase.supabase.storage.from_("products").download("faiss/search.faiss")
        with open(local_index_path, "wb") as f:
            f.write(index_data)
        print("✅ FAISS index file downloaded successfully")
        
        # Download metadata file
        print("📥 Downloading metadata file...")
        metadata_data = supabase.supabase.storage.from_("products").download("faiss/search_metadata.json")
        with open(local_metadata_path, "wb") as f:
            f.write(metadata_data)
        print("✅ Metadata file downloaded successfully")
        
        return True, "FAISS index files downloaded successfully"
        
    except Exception as e:
        error_message = f"Error downloading FAISS index files: {str(e)}"
        print(f"❌ {error_message}")
        return False, error_message 
