# Image Similarity Search Indexing Service

This project provides a backend service to build image similarity search indexes using the CLIP model for embeddings and FAISS for efficient searching. It can ingest images from PDF files or from a list of image URLs provided in a JSON file.

## Features

- **CLIP Model Embeddings:** Utilizes `openai/clip-vit-large-patch14` for generating rich image embeddings.
- **FAISS Indexing:** Employs FAISS for creating efficient similarity search indexes.
- **Multiple Input Sources:**
  - **PDF Processing:** Extracts images directly from uploaded PDF files.
  - **JSON (Image URLs):** Downloads images from URLs specified in a JSON payload.
- **Image Augmentation:** Applies a set of transformations to images before embedding to improve robustness. Embeddings from augmented variants are averaged.
- **Metadata Storage:** Saves metadata alongside the index, linking indexed images back to their original source (PDF page and image number, or original JSON data).
- **Configurable:** Settings like model name, file paths, and server configuration can be managed via `app/config.py`.

## Core Components

- `app/config.py`: Contains configuration variables for the application, including model details, file paths, and Flask server settings.
- `app/services/handle_build_index.py`: Implements the core logic for:
  - Extracting images from PDFs (`fitz`/PyMuPDF).
  - Downloading images from URLs.
  - Preprocessing images (transformations, augmentations).
  - Generating embeddings using the CLIP model.
  - Building and saving FAISS indexes (`search.faiss`).
  - Saving associated metadata (`search_metadata.json`).

## How It Works

The service exposes functionalities (likely through API endpoints, though not explicitly defined in the provided files) to trigger the index-building process.

1.  **Input:** The user provides either PDF files or a JSON structure containing image URLs.
2.  **Image Extraction/Downloading:**
    - For PDFs: Images are extracted page by page.
    - For JSON: Images are downloaded from the provided URLs.
3.  **Preprocessing & Embedding:**
    - Each image undergoes several augmentations (e.g., flips, rotations, color jitter).
    - The CLIP model generates an embedding for each augmented variant.
    - These variant embeddings are averaged to produce a single robust embedding for the original image.
4.  **Index Creation:**
    - The collected embeddings are used to build a FAISS index (`IndexFlatL2` is used).
5.  **Storage:**
    - The FAISS index is saved to a file (default: `faiss_indices/search.faiss`).
    - A corresponding JSON metadata file is saved (default: `faiss_indices/search_metadata.json`), allowing mapping from index entries back to source details.

## Configuration

Key configurations can be found in `app/config.py`:

- `MODEL_NAME`: The pre-trained CLIP model to use.
- `INDEX_DIR`: Directory to store the generated FAISS index and metadata files (default: `"faiss_indices"` in `handle_build_index.py`).
- `INDEX_FILENAME`: Name for the FAISS index file (default: `"search.faiss"`).
- `METADATA_FILENAME`: Name for the metadata file (default: `"search_metadata.json"`).
- `BATCH_SIZE`: Number of images to process in a single batch during embedding generation.
- Flask server settings (`DEBUG`, `HOST`, `PORT`).

## Usage (Conceptual)

While specific API endpoints are not detailed in the provided files, the typical workflow would involve:

1.  **Starting the Service:** Run the Flask application.
2.  **Building an Index:**
    - Send a request (e.g., POST) to an endpoint dedicated to PDF processing, uploading the PDF files.
    - Or, send a request to an endpoint for JSON processing, providing the JSON data with image URLs in the request body.
3.  **Receiving Output:** The service will respond with a success message, including the local paths to the generated FAISS index and metadata file, and the number of items indexed.

    Example response snippet from `handle_build_index.py`:

    ```json
    {
      "message": "Index built successfully...",
      "local_index_path": "faiss_indices/search.faiss",
      "local_metadata_path": "faiss_indices/search_metadata.json",
      "num_items_indexed": 123
    }
    ```

## Dependencies (inferred from `handle_build_index.py`)

- `torch`
- `faiss-cpu` (or `faiss-gpu` if CUDA is available and preferred)
- `Pillow` (PIL)
- `tqdm`
- `transformers`
- `torchvision`
- `numpy`
- `Flask` (for serving, indicated by `jsonify`)
- `requests` (for downloading images from URLs)
- `PyMuPDF` (fitz) (for PDF processing)

It's recommended to manage these dependencies using a `requirements.txt` file.

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a .env file):
   ```bash
   FLASK_DEBUG=True  # Set to False in production
   ```

5. Run the Flask application:
   ```bash
   python run.py
   ```
   Or using Gunicorn (production):
   ```bash
   gunicorn run:app
   ```

The service will be available at:
- Development: http://localhost:5001
- Production: Depends on your deployment configuration


---

This README provides a general overview based on the provided service logic. Specific API endpoint definitions, deployment instructions, and detailed usage examples would require further information about the application's routing and entry points.
