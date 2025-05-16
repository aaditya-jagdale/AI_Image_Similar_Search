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
successful_image_saves = 0 # Counter for successfully saved images

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

        # --- Field Extraction Logic (New Format) ---
        page_data = {}
        design_no_for_image = None
        label_index = -1

        # Find the index of "Design no:"
        try:
            label_index = lines.index("Design no:")
        except ValueError:
            # "Design no:" not found on this page
            # print(f"  Warning: Skipping page {i+1} in {os.path.basename(pdf_path)} because 'Design no:' label was not found.")
            continue # Skip to the next page

        # Check if the subsequent labels match and enough lines exist for values
        required_labels = ["Width:", "Stock:", "GSM:"]
        values_start_index = label_index + len(required_labels) + 1 # Index where values should start

        # Verify labels
        labels_match = True
        for k, label in enumerate(required_labels):
            current_label_index = label_index + 1 + k
            if current_label_index >= len(lines) or lines[current_label_index] != label:
                labels_match = False
                # print(f"  Warning: Skipping page {i+1} in {os.path.basename(pdf_path)} because expected label sequence was not found after 'Design no:'. Expected '{label}'.")
                break

        if not labels_match:
            continue # Skip to the next page

        # Try to extract Design No, Width, Stock, and conditionally GSM
        try:
            if values_start_index + 2 < len(lines): # Are there at least 3 value lines for DesignNo, Width, Stock?
                design_no = lines[values_start_index]
                width = lines[values_start_index + 1].replace('"', '')
                stock = lines[values_start_index + 2]

                # Now check for GSM value
                if values_start_index + 3 < len(lines): # Is there a 4th value line for GSM?
                    gsm = lines[values_start_index + 3]
                else: # Only 3 value lines found (DesignNo, Width, Stock), GSM is missing
                    gsm = "0" # Default GSM to "0"

                page_data["design no"] = design_no
                page_data["width"] = width
                page_data["stock"] = stock
                page_data["GSM"] = gsm
                design_no_for_image = design_no # Crucial for image naming

            else: # Not enough lines for even the first 3 values (Design No, Width, Stock)
                # print(f"  Warning: Skipping page {i+1} in {os.path.basename(pdf_path)} because not enough lines for Design No, Width, Stock after labels.")
                continue # Skip page
        except IndexError:
            # This might happen if values_start_index is somehow invalid despite earlier checks,
            # or if lines are shorter than expected in a way not caught by the if/else.
            print(f"  Warning: Skipping page {i+1} in {os.path.basename(pdf_path)} due to unexpected IndexError during value extraction.")
            continue # Skip page
        # --- End of Field Extraction Logic ---
        
        # Skip if essential data wasn't found (design_no is key)
        if not design_no_for_image:
            # This case should ideally be caught earlier, but added as a safeguard
            continue

        # --- Image Extraction Logic ---
        image_path_for_json = None # Path to store in JSON
        try:
            image_list = page.get_images(full=True)
            print(f"  Page {i+1} in {os.path.basename(pdf_path)} (Design No: {design_no_for_image}): Found {len(image_list)} image(s).")

            if image_list:
                best_pix = None
                best_xref = None
                max_area = -1

                for img_info in image_list:
                    xref = img_info[0]
                    current_pix = None
                    try:
                        current_pix = fitz.Pixmap(doc, xref)
                        print(f"    Checking image xref {xref}: width={current_pix.width}, height={current_pix.height}, colorspace={current_pix.colorspace.name}, area={current_pix.width * current_pix.height}")

                        # Skip tiny images, often icons or noise
                        if current_pix.width < 50 or current_pix.height < 50:
                            print(f"      Skipping xref {xref} as it's too small ({current_pix.width}x{current_pix.height}).")
                            if current_pix:
                                current_pix = None # Release Pixmap
                            continue

                        area = current_pix.width * current_pix.height
                        if area > max_area:
                            if best_pix: # Release previous best_pix
                                best_pix = None
                            max_area = area
                            best_pix = current_pix
                            best_xref = xref
                            # Don't release current_pix here, it's now best_pix
                        else:
                            if current_pix: # Release if not chosen and not best_pix
                                current_pix = None
                    
                    except Exception as e_pix:
                        print(f"    Error processing image xref {xref} on page {i+1}: {e_pix}")
                        if current_pix:
                             current_pix = None # Release Pixmap
                        continue # Try next image in list

                if best_pix:
                    print(f"    Selected image xref {best_xref} with largest area {max_area} for Design No: {design_no_for_image}")
                    # Clean design number for use in filename (replace / and invalid chars)
                    safe_design_no = re.sub(r'[^a-zA-Z0-9_-]+', '_', design_no_for_image.replace('/', '-'))
                    image_save_path = os.path.join(output_folder, f"{safe_design_no}.png")
                    try:
                        best_pix.save(image_save_path)
                        image_path_for_json = image_save_path
                        successful_image_saves += 1
                        print(f"    Successfully saved image for {safe_design_no} to {image_save_path}")
                    except Exception as e_save:
                        print(f"    Error saving selected image for Design No: {design_no_for_image} (xref: {best_xref}): {e_save}")
                    finally:
                        if best_pix: # Release the chosen Pixmap after saving
                            best_pix = None
                else:
                    print(f"  No suitable image found on page {i+1} for Design No: {design_no_for_image} after filtering.")

            else: # No images found on page
                print(f"  No images found on page {i+1} in {os.path.basename(pdf_path)} for Design No: {design_no_for_image}.")

        except Exception as e_outer:
             print(f"  Error during image extraction process for page {i+1} (Design No: {design_no_for_image}) from {os.path.basename(pdf_path)}: {e_outer}")

        # --- Build record and append to main data list ---
        # Use the exact keys requested by the user
        record = {
            "design no": page_data.get("design no"),
            "width": page_data.get("width"),
            "stock": page_data.get("stock"),
            "GSM": page_data.get("GSM"), # Changed key to "GSM"
            "image": image_path_for_json,
            "source_pdf": os.path.basename(pdf_path)
        }
        # Only append if we actually got the core data (design_no, width, stock, GSM)
        if record["design no"] and record["width"] and record["stock"] and record["GSM"]:
             data.append(record)
             
    # Close the document after processing all its pages
    doc.close()

# --- Print Stats & Save Combined JSON ---
print(f"\n--- Extraction Stats ---")
print(f"Successfully saved images: {successful_image_saves}")
print(f"Total records extracted: {len(data)}")
print(f"Records without saved images: {len(data) - successful_image_saves}")
print(f"------------------------")

if data:
    print(f"\nSaving combined data from {len(data)} records to {output_json_path}...")
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Extraction complete.")
else:
    print("\nNo data extracted from any PDF files.")
