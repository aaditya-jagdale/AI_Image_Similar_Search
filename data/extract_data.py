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
# We'll create simple print fallbacks if it can't be imported
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
# NEW: Directory for images that are likely placeholders or "default images"
SUSPICIOUS_IMAGE_DIR = OUTPUT_DIR / "suspicious_images"
MASTER_JSON_FILE = OUTPUT_DIR / "pdf_extracted_data.json"
LIMIT = 100  # Set to None to process all files, or a number to limit
# Toggle GSM extraction: True = extract GSM field, False = ignore GSM field
EXTRACT_GSM = True

# NEW: Minimum area (width * height) for an image to be considered "valid"
# Anything smaller will be flagged as suspicious. 50000 = ~224x224 pixels
MIN_IMAGE_AREA = 50000  

# --- REGEX PATTERNS (compiled for efficiency) ---
# (No changes to regex)
RE_DESIGN_NO = re.compile(r"Design no:\s*\n\s*([^\n]+)", re.IGNORECASE | re.DOTALL)
RE_WIDTH = re.compile(r"Width:\s*\n\s*([^\n]+)", re.IGNORECASE | re.DOTALL)
RE_STOCK = re.compile(r"Stock:\s*\n\s*([^\n]+)", re.IGNORECASE | re.DOTALL)
RE_GSM = re.compile(r"GSM:\s*\n\s*([^\n]*)", re.IGNORECASE | re.DOTALL)


def setup_directories():
    """Ensures all necessary output directories exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)
    SKIPPED_IMAGE_DIR.mkdir(exist_ok=True)
    SUSPICIOUS_IMAGE_DIR.mkdir(exist_ok=True)  # NEW: Create suspicious dir
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


def clean_value(value):
    """Helper function to strip whitespace and remove extra characters."""
    if value:
        # Remove common OCR artifacts like '$'
        return value.strip().replace('$', '')
    return None


def clean_numeric_value(value):
    """Helper function to extract and convert numeric values to integers."""
    if value:
        # Remove common OCR artifacts and non-numeric characters except digits
        cleaned = value.strip().replace('$', '').replace('"', '').replace("'", '')
        # Extract only digits
        digits = ''.join(filter(str.isdigit, cleaned))
        if digits:
            return int(digits)
    return None


def get_regex_match(pattern, text):
    """Finds a regex match and returns the cleaned group 1, or None."""
    match = pattern.search(text)
    if match:
        return clean_value(match.group(1))
    return None


def sanitize_filename(name):
    """Replaces characters that are unsafe for filenames."""
    if not name:
        return None
    return name.replace('/', '-').replace('\\', '-')


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
    pdf_filename_stem = pdf_path.stem  # e.g., "CHAMBRAY"

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        error(f"Error opening {pdf_path.name}: {e}")
        return []

    # Iterate through pages, skipping the first (index 0) which is the title
    for page_num in range(1, len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        # --- Text Extraction Logic (Unchanged) ---
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        design_no = None
        width = None
        stock = None
        gsm = None
        
        label_indices = {}
        for i, line in enumerate(lines):
            if line == 'Design no:' or line.startswith('Design no:'):
                label_indices['design_no'] = i
            elif line == 'Width:' or line.startswith('Width:'):
                label_indices['width'] = i
            elif line == 'Stock:' or line.startswith('Stock:'):
                label_indices['stock'] = i
            elif (line == 'GSM:' or line.startswith('GSM:')) and EXTRACT_GSM:
                label_indices['gsm'] = i
        
        required_labels = 4 if EXTRACT_GSM else 3
        
        if len(label_indices) == required_labels:
            last_label_idx = max(label_indices.values())
            values_start_idx = last_label_idx + 1
            
            if EXTRACT_GSM:
                if values_start_idx + 3 < len(lines):
                    design_no = clean_value(lines[values_start_idx])
                    width = clean_numeric_value(lines[values_start_idx + 1])
                    stock = clean_numeric_value(lines[values_start_idx + 2])
                    gsm = clean_numeric_value(lines[values_start_idx + 3])
            else:
                if values_start_idx + 2 < len(lines):
                    design_no = clean_value(lines[values_start_idx])
                    width = clean_numeric_value(lines[values_start_idx + 1])
                    stock = clean_numeric_value(lines[values_start_idx + 2])
        
        if not design_no:
            info(f"  Skipping page {page_num + 1} (no Design No. found) - saving image to skipped folder.")
            
            # (Unchanged logic for skipped pages)
            image_list = page.get_images(full=True)
            if image_list:
                try:
                    xref = image_list[0][0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = f"{pdf_filename_stem}_page{page_num + 1}.{image_ext}"
                    image_save_path = SKIPPED_IMAGE_DIR / image_filename
                    with open(image_save_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    success(f"    → Saved skipped image: {image_filename}")
                except Exception as e:
                    error(f"    Error extracting image from skipped page {page_num + 1}: {e}")
            
            continue

        # --- NEW Image Extraction Logic ---
        image_path_str = None
        image_status = "no_image_found_on_page" # Default status
        
        image_list = page.get_images(full=True)
        
        if image_list:
            largest_image_info = None
            max_area = 0
            
            # Loop through all images to find the largest one
            for img_info in image_list:
                # img_info is [xref, smask, width, height, ...]
                width = img_info[2]
                height = img_info[3]
                area = width * height
                
                if area > max_area:
                    max_area = area
                    largest_image_info = img_info
            
            if largest_image_info:
                try:
                    xref = largest_image_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    safe_design_no = sanitize_filename(design_no)
                    image_filename = f"{pdf_filename_stem}_{safe_design_no}.{image_ext}"
                    
                    # Check if the largest image is still too small
                    if max_area < MIN_IMAGE_AREA:
                        # This is likely a "default image" or placeholder
                        image_save_path = SUSPICIOUS_IMAGE_DIR / image_filename
                        image_status = f"suspicious_small_image (Area: {max_area})"
                        info(f"  Found small image for {design_no}, saving to suspicious folder.")
                    else:
                        # This is a good image
                        image_save_path = IMAGE_DIR / image_filename
                        image_status = f"extracted (Area: {max_area})"
                    
                    # Save the image
                    with open(image_save_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_path_str = str(image_save_path.relative_to(OUTPUT_DIR))

                except Exception as e:
                    error(f"  Error extracting largest image on page {page_num + 1}: {e}")
                    image_status = f"extraction_error: {e}"
            else:
                image_status = "no_image_found_in_list"
        
        # Create the data object for this page
        entry = {
            "source_pdf": pdf_path.name,
            "design_no": design_no,
            "width": width,
            "stock": stock,
            "image_path": image_path_str,
            "image_status": image_status  # NEW: Add status field
        }
        
        if EXTRACT_GSM:
            entry["gsm"] = gsm
            
        new_entries.append(entry)
        if design_no:
             success(f"  Extracted: {design_no} (Image: {image_status})")

    doc.close()
    return new_entries


def main():
    """Main execution function."""
    setup_directories()
    
    pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))
    if not pdf_files:
        error(f"No PDF files found in {PDF_DIRECTORY}. Exiting.")
        info("Please add your PDFs (like CHAMBRAY.pdf) to that folder.")
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
    success(f"Suspicious/Default images saved to {SUSPICIOUS_IMAGE_DIR}")


if __name__ == "__main__":
    main()