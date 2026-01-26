import fitz  # PyMuPDF
import re
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for 'app' package imports
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Assuming 'app.core.console' is a custom module for logging
try:
    from app.core.console import success, error, info
except ImportError:
    print("app.core.console not found. Using simple print statements.")
    def success(msg): print(f"SUCCESS: {msg}")
    def error(msg): print(f"ERROR: {msg}")
    def info(msg): print(f"INFO: {msg}")

# --- CONSTANTS ---
PDF_DIRECTORY = Path("data/pdfs")
OUTPUT_DIR = Path("data/output")
IMAGE_DIR = OUTPUT_DIR / "images"
SKIPPED_IMAGE_DIR = OUTPUT_DIR / "skipped_images"
MASTER_JSON_FILE = OUTPUT_DIR / "pdf_extracted_data.json"
LIMIT = None  # Set to None to process all files, or a number to limit

# Minimum image area to consider valid (50000 = ~224x224 pixels)
MIN_IMAGE_AREA = 50000

def setup_directories():
    """Ensures all necessary output directories exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)
    SKIPPED_IMAGE_DIR.mkdir(exist_ok=True)
    # Ensure the master JSON file exists
    if not MASTER_JSON_FILE.exists():
        save_master_data([])


def load_master_data():
    """Loads the master JSON data file safely."""
    try:
        with open(MASTER_JSON_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_master_data(data):
    """Saves data to the master JSON file."""
    with open(MASTER_JSON_FILE, 'w') as f:
        json.dump(data, f, indent=4)


def clean_text_value(value):
    """
    Clean text values by removing extra whitespace and common OCR artifacts.
    Preserves the original format (like 61" for width).
    """
    if value:
        return value.strip().replace('$', '')
    return None


def extract_numeric_value(value):
    """
    Extract numeric value from text.
    Returns integer if found, otherwise 0.
    """
    if not value:
        return 0
    
    # Remove common OCR artifacts
    cleaned = value.strip().replace('$', '').replace('"', '').replace("'", '')
    # Extract only digits
    digits = ''.join(filter(str.isdigit, cleaned))
    
    if digits:
        return int(digits)
    return 0


def sanitize_filename(name):
    """Replaces characters that are unsafe for filenames."""
    if not name:
        return None
    return name.replace('/', '-').replace('\\', '-').replace(' ', '_')


def extract_data_from_page(text):
    """
    Extracts Design no, Width, Stock, and GSM from page text.
    
    Returns:
        dict: Contains design_no, width, stock, gsm or None if extraction fails
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Find label positions
    label_indices = {}
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if 'design no:' in line_lower or line == 'Design no:':
            label_indices['design_no'] = i
        elif 'width:' in line_lower or line == 'Width:':
            label_indices['width'] = i
        elif 'stock:' in line_lower or line == 'Stock:':
            label_indices['stock'] = i
        elif 'gsm:' in line_lower or line == 'GSM:':
            label_indices['gsm'] = i
    
    # Need at least Design no, Width, and Stock
    if len(label_indices) < 3:
        return None
    
    # Find where values start (after the last label)
    last_label_idx = max(label_indices.values())
    values_start_idx = last_label_idx + 1
    
    # Check if we have enough lines for values
    if values_start_idx >= len(lines):
        return None
    
    # Extract values based on label order
    design_no = None
    width = None
    stock = None
    gsm = 0  # Default to 0 if missing
    
    # Sort labels by their index to get the correct order
    sorted_labels = sorted(label_indices.items(), key=lambda x: x[1])
    
    # Extract values in order
    for idx, (label_name, label_idx) in enumerate(sorted_labels):
        value_idx = values_start_idx + idx
        if value_idx < len(lines):
            value = lines[value_idx]
            
            if label_name == 'design_no':
                design_no = clean_text_value(value)
            elif label_name == 'width':
                width = clean_text_value(value)  # Keep as text with "
            elif label_name == 'stock':
                stock = extract_numeric_value(value)
            elif label_name == 'gsm':
                gsm = extract_numeric_value(value)
    
    # If GSM wasn't found in labels, try to get it from the value after stock
    if 'gsm' not in label_indices and values_start_idx + 3 < len(lines):
        gsm = extract_numeric_value(lines[values_start_idx + 3])
    
    if not design_no:
        return None
    
    return {
        'design_no': design_no,
        'width': width,
        'stock': stock,
        'gsm': gsm
    }


def extract_largest_image(page, doc):
    """
    Extracts the largest image from a page.
    
    Returns:
        tuple: (image_bytes, image_ext, area) or (None, None, 0) if no valid image
    """
    image_list = page.get_images(full=True)
    
    if not image_list:
        return None, None, 0
    
    largest_image_info = None
    max_area = 0
    
    # Find the largest image by area
    for img_info in image_list:
        width = img_info[2]
        height = img_info[3]
        area = width * height
        
        if area > max_area:
            max_area = area
            largest_image_info = img_info
    
    if not largest_image_info or max_area < MIN_IMAGE_AREA:
        return None, None, max_area
    
    try:
        xref = largest_image_info[0]
        base_image = doc.extract_image(xref)
        return base_image["image"], base_image["ext"], max_area
    except Exception as e:
        error(f"Error extracting image: {e}")
        return None, None, max_area


def process_pdf(pdf_path):
    """
    Extracts all required data from a single PDF file.
    
    Args:
        pdf_path (Path): The path to the PDF file to process.

    Returns:
        list: A list of dictionaries, one for each data page in the PDF.
    """
    info(f"Processing {pdf_path.name}...")
    new_entries = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        error(f"Error opening {pdf_path.name}: {e}")
        return []

    # Iterate through pages, skipping the first (index 0) which is the title
    for page_num in range(1, len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        # Extract text data
        data = extract_data_from_page(text)
        
        if not data:
            info(f"  Skipping page {page_num + 1} (no valid data found)")
            
            # Save image from skipped page
            image_bytes, image_ext, area = extract_largest_image(page, doc)
            if image_bytes:
                image_filename = f"{pdf_filename_stem}_page{page_num + 1}.{image_ext}"
                image_save_path = SKIPPED_IMAGE_DIR / image_filename
                with open(image_save_path, "wb") as img_file:
                    img_file.write(image_bytes)
                success(f"    → Saved skipped image: {image_filename}")
            
            continue

        # Extract image
        image_path_str = None
        image_status = "no_image_found"
        
        image_bytes, image_ext, area = extract_largest_image(page, doc)
        
        if image_bytes:
            safe_design_no = sanitize_filename(data['design_no'])
            image_filename = f"{safe_design_no}.{image_ext}"
            image_save_path = IMAGE_DIR / image_filename
            
            with open(image_save_path, "wb") as img_file:
                img_file.write(image_bytes)
            
            image_path_str = str(image_save_path.relative_to(OUTPUT_DIR))
            image_status = f"extracted (Area: {area})"
            success(f"  Extracted image: {image_filename}")
        elif area > 0:
            image_status = f"image_too_small (Area: {area})"
        
        # Create entry
        entry = {
            "source_pdf": pdf_path.name,
            "design_no": data['design_no'],
            "width": data['width'],
            "stock": data['stock'],
            "gsm": data['gsm'],
            "image_path": image_path_str,
            "image_status": image_status
        }
        
        new_entries.append(entry)
        success(f"  Extracted: {data['design_no']} | Width: {data['width']} | Stock: {data['stock']} | GSM: {data['gsm']}")

    doc.close()
    return new_entries


def main():
    """Main execution function."""
    setup_directories()
    
    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))
    if not pdf_files:
        error(f"No PDF files found in {PDF_DIRECTORY}. Exiting.")
        info("Please add your PDFs to that folder.")
        return

    master_data = load_master_data()
    
    existing_entries = {
        (entry.get("source_pdf"), entry.get("design_no")) 
        for entry in master_data
    }
    
    process_limit = pdf_files
    if LIMIT is not None:
        process_limit = pdf_files[:LIMIT]
        info(f"Processing limited to {LIMIT} files.")

    for pdf_path in process_limit:
        new_data_from_pdf = process_pdf(pdf_path)
        
        for entry in new_data_from_pdf:
            entry_key = (entry.get("source_pdf"), entry.get("design_no"))
            if entry_key not in existing_entries:
                master_data.append(entry)
                existing_entries.add(entry_key)
            else:
                info(f"  Skipping duplicate entry: {entry_key}")

    save_master_data(master_data)
    success("\nProcessing complete.")
    success(f"Master data saved to {MASTER_JSON_FILE}")
    success(f"Images saved to {IMAGE_DIR}")
    success(f"Skipped page images saved to {SKIPPED_IMAGE_DIR}")
    info(f"Total entries: {len(master_data)}")


if __name__ == "__main__":
    main()