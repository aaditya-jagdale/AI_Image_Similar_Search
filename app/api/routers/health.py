from fastapi import APIRouter
router = APIRouter()

#health end point
@router.get("/healthz")
def read_root():
    return {"message": "ok"}
