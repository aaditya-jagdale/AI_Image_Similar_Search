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
    
    # ------------------------- Upload Data -------------------------
    def upload_data(self, image:str, design_no:str, width:str, stock:str, GSM:str, source_pdf:str, source_url:str):
        """Uploads data to the 'telegram_data' table in Supabase."""
        try:
            response = self.supabase.table("telegram_data").insert({
                "image": image,
                "design_no": design_no,
                "width": int(width),
                "stock": int(stock),
                "GSM": int(GSM),
                "source_pdf": source_pdf,
                "source_url": source_url
            }).execute()
            print(f"Successfully uploaded data to Supabase")
            return response
        except Exception as e:
            print(f"Error uploading data to Supabase: {e}")
            return None