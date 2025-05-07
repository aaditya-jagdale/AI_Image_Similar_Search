import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

class SupabaseService:
    def __init__(self):
        self.supabase = create_client(
            supabase_url= os.getenv("SUPABASE_URL"),
            supabase_key= os.getenv("SUPABASE_ANON_KEY")
        )
    
    def upload_faiss(self, local_file_path: str, supabase_path: str):
        """Uploads a file to the 'faiss' bucket in Supabase Storage."""
        try:
            with open(local_file_path, 'rb') as f: # Open in binary read mode
                response = (
                    self.supabase.storage  # Correct: Use self.supabase
                    .from_("faiss")
                    .upload(
                        path=supabase_path, # Destination path in Supabase
                        file=f,             # File object to upload
                        file_options={"upsert": "true"} # Overwrite if exists
                    )
                )
            print(f"Successfully uploaded {local_file_path} to Supabase at {supabase_path}")
            return response
        except Exception as e:
            print(f"Error uploading {local_file_path} to Supabase: {e}")
            # Consider how to handle upload errors (e.g., raise exception, return None)
            return None