from fastapi import Header, HTTPException
from app.search import validate_token

async def get_current_user(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    user_id = validate_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id
