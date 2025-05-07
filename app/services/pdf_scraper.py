import fitz
import json
import os
import re
import glob # Import glob to find PDF files

# --- Input and Output Configuration ---
pdf_folder = "pdfs" # Folder containing the PDF files
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)
output_json_path = "output.json" # Path for the final JSON output
data = [] # Initialize the list to hold data from ALL PDFs

# --- Find all PDF files ---
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

if not pdf_files:
    print(f"No PDF files found in the '{pdf_folder}' directory.")
    exit()

print(f"Found {len(pdf_files)} PDF files to process:")
for pdf_path in pdf_files:
    print(f"- {os.path.basename(pdf_path)}")

# --- Process Each PDF File ---
for pdf_path in pdf_files:
    print(f"\nProcessing {os.path.basename(pdf_path)}...")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {os.path.basename(pdf_path)}: {e}. Skipping this file.")
        continue
        
    # Extract a prefix from the PDF filename for image naming uniqueness
    pdf_filename_base = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_prefix = re.sub(r'\W+', '_', pdf_filename_base) # Clean filename for use as prefix

    # --- Loop through pages (starting from page 2) ---
    for i in range(1, len(doc)):
        page = doc[i]
        try:
            text = page.get_text()
        except Exception as e:
            print(f"  Error getting text from page {i+1} in {os.path.basename(pdf_path)}: {e}. Skipping page.")
            continue
            
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # --- Field Extraction Logic ---
        page_data = {}
        design_no_for_image = None
        stock_label_index = -1

        for j, line in enumerate(lines):
            if line.lower() == "stock:":
                stock_label_index = j
                break

        if stock_label_index != -1:
            try:
                if stock_label_index + 3 < len(lines):
                    page_data["design_no"] = lines[stock_label_index + 1]
                    page_data["width"] = lines[stock_label_index + 2].replace('"', '')
                    page_data["stock"] = lines[stock_label_index + 3]
                    design_no_for_image = page_data["design_no"]
                else:
                    # print(f"  Warning: Skipping page {i+1} in {os.path.basename(pdf_path)} because expected values after 'Stock:' label were not found.")
                    continue
            except IndexError:
                # print(f"  Warning: Skipping page {i+1} in {os.path.basename(pdf_path)} due to parsing error near 'Stock:' label (IndexError).")
                continue
        else:
            # print(f"  Warning: Skipping page {i+1} in {os.path.basename(pdf_path)} because 'Stock:' label was not found.")
            continue
        # --- End of Field Extraction Logic ---
        
        # Skip if essential data wasn't found
        if not design_no_for_image:
            continue

        # --- Image Extraction Logic ---
        image_path_for_json = None # Path to store in JSON
        try:
            image_list = page.get_images(full=True)
            if image_list:
                xref = image_list[0][0]
                pix = fitz.Pixmap(doc, xref)
                safe_design_no = re.sub(r'[^a-zA-Z0-9_-]+', '', design_no_for_image.replace('/', '-'))
                # Add PDF prefix to image filename for uniqueness
                img_name = f"{pdf_prefix}_{safe_design_no}.png"
                # image_save_path = os.path.join(output_folder, img_name)
                image_save_path = f"images/{safe_design_no}.png"
                try:
                    pix.save(image_save_path)
                    image_path_for_json = image_save_path # Store the relative path
                except Exception as e:
                    print(f"  Error saving image for page {i+1} (design_no: {design_no_for_image}) from {os.path.basename(pdf_path)}: {e}")
        except Exception as e:
             print(f"  Error extracting image for page {i+1} (design_no: {design_no_for_image}) from {os.path.basename(pdf_path)}: {e}")

        # --- Build record and append to main data list ---
        record = {
            "design_no": page_data.get("design_no"),
            "width": page_data.get("width"),
            "stock": page_data.get("stock"),
            "image": image_path_for_json, # Use the path saved earlier
            "source_pdf": os.path.basename(pdf_path) # Add source PDF info
        }
        # Only append if we actually got the core data
        if record["design_no"] and record["width"] and record["stock"]:
             data.append(record)
             
    # Close the document after processing all its pages
    doc.close()

# --- Save Combined JSON ---
if data:
    print(f"\nSaving combined data from {len(data)} records to {output_json_path}...")
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Extraction complete.")
else:
    print("\nNo data extracted from any PDF files.")
