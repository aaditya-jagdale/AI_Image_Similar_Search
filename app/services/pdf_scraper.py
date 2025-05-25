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
pdf_processing_summary = [] # To store summary data for each PDF

# --- Find all PDF files ---
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

if not pdf_files:
    print(f"No PDF files found in the '{pdf_folder}' directory.")
    exit()
total_pages = 0
print(f"\nFound {len(pdf_files)} PDF files to process:")
print("\nIndividual PDF Files:")
print("-" * 60)
print(f"{'Filename':<30} {'Pages':>10}")
print("-" * 60)
for pdf_path in pdf_files:
    pages = len(fitz.open(pdf_path))
    total_pages += pages
    print(f"{os.path.basename(pdf_path):<30} {pages:>10}")
print("-" * 60)
print(f"{'Total Pages Across All PDFs:':>30} {total_pages:>10}")
print("-" * 60)


# --- Process Each PDF File ---
for pdf_path in pdf_files:
    pdf_name_for_summary = os.path.basename(pdf_path)
    successful_images_for_this_pdf = 0
    pages_processed_for_this_pdf = 0
    doc = None  # Initialize doc to None for the finally block

    print(f"\nProcessing {pdf_name_for_summary}...")
    try:
        doc = fitz.open(pdf_path)
        
        num_doc_pages = len(doc)
        pages_processed_for_this_pdf = max(0, num_doc_pages - 1) # Pages from 2nd onwards
        
        # Extract a prefix from the PDF filename for image naming uniqueness
        pdf_filename_base = os.path.splitext(pdf_name_for_summary)[0]
        pdf_prefix = re.sub(r'\W+', '_', pdf_filename_base) # Clean filename for use as prefix

        # --- Loop through pages (starting from page 2) ---
        for i in range(1, num_doc_pages):
            page = doc[i]
            try:
                text = page.get_text()
            except Exception as e:
                print(f"  Error getting text from page {i+1} in {pdf_name_for_summary}: {e}. Skipping page.")
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
                # print(f"  Warning: Skipping page {i+1} in {pdf_name_for_summary} because 'Design no:' label was not found.")
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
                    # print(f"  Warning: Skipping page {i+1} in {pdf_name_for_summary} because expected label sequence was not found after 'Design no:'. Expected '{label}'.")
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
                    # print(f"  Warning: Skipping page {i+1} in {pdf_name_for_summary} because not enough lines for Design No, Width, Stock after labels.")
                    continue # Skip page
            except IndexError:
                # This might happen if values_start_index is somehow invalid despite earlier checks,
                # or if lines are shorter than expected in a way not caught by the if/else.
                print(f"  Warning: Skipping page {i+1} in {pdf_name_for_summary} due to unexpected IndexError during value extraction.")
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

                if image_list:
                    best_pix = None
                    best_xref = None
                    max_area = -1

                    for img_info in image_list:
                        xref = img_info[0]
                        current_pix = None
                        try:
                            current_pix = fitz.Pixmap(doc, xref)

                            # Skip tiny images, often icons or noise
                            if current_pix.width < 50 or current_pix.height < 50:
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
                        # Clean design number for use in filename (replace / and invalid chars)
                        safe_design_no = re.sub(r'[^a-zA-Z0-9_-]+', '_', design_no_for_image.replace('/', '-'))
                        image_save_path = os.path.join(output_folder, f"{safe_design_no}.png")
                        try:
                            # --- Resize the image --- 
                            new_width = int(best_pix.width * 0.3) # 30% of original width
                            new_height = int(best_pix.height * 0.3) # 30% of original height
                            
                            # Create a new, scaled pixmap
                            # The constructor Pixmap(source, width, height) handles scaling.
                            if new_width > 0 and new_height > 0:
                                scaled_pix = fitz.Pixmap(best_pix, new_width, new_height)
                                scaled_pix.save(image_save_path)
                                if scaled_pix: # Release the scaled pixmap
                                    scaled_pix = None
                            else: # If scaling results in zero dimension, save original (or handle as error)
                                print(f"    Warning: Scaled dimensions for {safe_design_no} are too small ({new_width}x{new_height}). Saving original.")
                                best_pix.save(image_save_path)
                            # --- End Resize --- 
                            
                            image_path_for_json = image_save_path
                            successful_image_saves += 1
                            successful_images_for_this_pdf += 1 # Count for this PDF
                        except Exception as e_save:
                            print(f"    Error saving selected image for Design No: {design_no_for_image} (xref: {best_xref}): {e_save}")
                        finally:
                            if best_pix: # Release the chosen Pixmap after saving
                                best_pix = None
                    else:
                        print(f"  No suitable image found on page {i+1} for Design No: {design_no_for_image} after filtering.")

                else: # No images found on page
                    print(f"  No images found on page {i+1} in {pdf_name_for_summary} for Design No: {design_no_for_image}.")

            except Exception as e_outer:
                 print(f"  Error during image extraction process for page {i+1} (Design No: {design_no_for_image}) from {pdf_name_for_summary}: {e_outer}")

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
                 
    except Exception as e: # This except block pairs with the try: doc = fitz.open(pdf_path)
        print(f"Error opening or processing {pdf_name_for_summary}: {e}. Skipping this file or parts of it.")
        # If doc could not be opened, num_doc_pages might not be set. 
        # We still want to record this attempt in the summary.
        # pages_processed_for_this_pdf will be 0 if len(doc) failed, or max(0, num_doc_pages - 1) if it opened but failed later.

    finally:
        if doc: # Ensure doc is closed if it was successfully opened
            doc.close()
        
        # Append summary for this PDF regardless of success or failure during its processing
        pdf_processing_summary.append({
            "pdf_name": pdf_name_for_summary,
            "total_pages_processed": pages_processed_for_this_pdf, # This reflects pages *attempted* for data extraction
            "successful_images": successful_images_for_this_pdf
        })

# --- Print PDF Processing Summary Table ---
print("\n\n--- PDF Processing Summary ---")
print("-" * 100)
print(f"{'PDF Name':<40} {'Pages Processed':<20} {'Successful Images':<20} {'Failed Images':>15}")
print("-" * 100)
for summary_item in pdf_processing_summary:
    failed = summary_item['total_pages_processed'] - summary_item['successful_images']
    print(f"{summary_item['pdf_name']:<40} {summary_item['total_pages_processed']:<20} {summary_item['successful_images']:<20} {failed:>15}")
print("-" * 100)

# --- Print Stats & Save Combined JSON ---
print(f"\n--- Extraction Stats ---")
print(f"Successfully saved images: {successful_image_saves}")
print(f"Total records extracted: {len(data)}")
print(f"Records without saved images: {len(data) - successful_image_saves}")
print(f"Failed image extractions: {sum(s['total_pages_processed'] - s['successful_images'] for s in pdf_processing_summary)}")
print(f"------------------------")

if data:
    print(f"\nSaving combined data from {len(data)} records to {output_json_path}...")
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Extraction complete.")
else:
    print("\nNo data extracted from any PDF files.")
