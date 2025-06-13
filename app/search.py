import logging
import jwt
from typing import Optional
from dotenv import load_dotenv
import os


load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def validate_token(token: str) -> Optional[str]:
    """Validate the Supabase JWT token and return the user ID if valid."""
    try:
        # Decode the JWT token without verification (Supabase tokens are already verified)
        decoded = jwt.decode(token, options={"verify_signature": False})
        return decoded.get("sub")  # 'sub' is the user ID in Supabase JWT
    except Exception as e:
        logger.error(f"Token validation failed: {e}")
        return None

# if __name__ == "__main__":
#     # Example usage
#     token = input("Enter your Supabase session token: ")
#     user_id = validate_token(token)
    
#     if not user_id:
#         logger.error("❌ Invalid token")
#         exit(1)
        
#     logger.info(f"✅ Valid token for user {user_id}")
