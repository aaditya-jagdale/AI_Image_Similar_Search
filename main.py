import io
import os
from PIL import Image
from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from search_image import get_image_embedding

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
def test_page():
    """Serve the tester UI from the API origin so fetch() is same-origin (no CORS)."""
    return FileResponse(os.path.join(_ROOT, "test_api.html"))

base_router = APIRouter(prefix="/api/v1")

@base_router.get("/")
def root_fun():
    return {
        "status":"okay"
    }

# take image as an input (POST + multipart; browsers cannot upload files on GET)
@base_router.post("/search")
async def search_image(image_file: UploadFile = File()):
    contents = await image_file.read()          # read bytes
    image = Image.open(io.BytesIO(contents))
    result = await get_image_embedding(image)
    output = []
    for id, distance, metadata in zip(result['ids'][0], result['distances'][0] ,result['metadatas'][0]):
        output.append({
                "id": id,
                "distance": distance,
                "metadata": metadata
            })
    return output


app.include_router(base_router)

_media_root = os.path.join(_ROOT, "data", "output")
if os.path.isdir(_media_root):
    app.mount("/media", StaticFiles(directory=_media_root), name="media")