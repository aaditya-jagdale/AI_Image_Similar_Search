import json
import pandas as pd
import os
import base64 # Added for image embedding
# We will use XlsxWriter as the engine for pandas to enable image embedding.

def get_image_base64(image_path):
    """Converts an image to a base64 string."""
    if not image_path or not isinstance(image_path, str) or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image {image_path} for base64 encoding: {e}")
        return None

def organize_data_to_html(json_file_path="output.json", html_file_path="organized_pdf_data.html"):
    """
    Reads data from a JSON file, organizes it into an HTML table, 
    optionally embeds images as base64, and saves it to an HTML file.

    Args:
        json_file_path (str): Path to the input JSON file.
        html_file_path (str): Path to the output HTML file.
    """
    # --- USER TOGGLE --- 
    # Set this to False to exclude the actual image embedding and "Image" column for testing
    include_actual_images = True 
    # ------------------

    # --- Path for the statistics file ---
    stats_file_path = "processing_stats.json" # Assuming it's in the same directory or adjust as needed
    # ------------------------------------

    try:
        # --- Read statistics data ---
        stats_data = None
        if os.path.exists(stats_file_path):
            with open(stats_file_path, 'r') as f_stats:
                stats_data = json.load(f_stats)
        else:
            print(f"Warning: Statistics file not found at {stats_file_path}")
        # --------------------------

        if not os.path.exists(json_file_path):
            print(f"Error: JSON file not found at {json_file_path}")
            return

        with open(json_file_path, 'r') as f:
            data = json.load(f)

        if not data:
            print("No data found in the JSON file.")
            return

        df = pd.DataFrame(data)
        # num_rows = len(df) if not df.empty else 0 # Not strictly needed for HTML like this

        column_mapping = {
            "source_pdf": "PDF name",
            "design no": "Design no",
            "width": "Width",
            "stock": "Stock",
            "GSM": "GSM",
            "image": "Image Path"  # Original image path from JSON
        }
        df.rename(columns=column_mapping, inplace=True)

        # Define base columns (always included)
        base_columns = ["PDF name", "Design no", "Width", "Stock", "GSM"]
        if include_actual_images:
            base_columns.append("Image")
        else:
            base_columns.append("Image Path")

        # Start HTML content - using a list of strings to join later for cleaner construction
        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html lang=\"en\">")
        html_parts.append("<head>")
        html_parts.append("    <meta charset=\"UTF-8\">")
        html_parts.append("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">")
        html_parts.append("    <title>Organized PDF Data</title>")
        html_parts.append("    <style>")
        html_parts.append("        body { font-family: Arial, sans-serif; margin: 20px; }")
        html_parts.append("        table { border-collapse: collapse; width: 100%; border: 1px solid #ddd; margin-bottom: 20px; }") # Added margin-bottom
        html_parts.append("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: middle; }")
        html_parts.append("        th { background-color: #f2f2f2; font-weight: bold; }")
        # Adjusted image and cell sizes
        html_parts.append("        img { max-width: 400px; max-height: 300px; display: block; margin: auto; }") 
        html_parts.append("        td.image-cell { width: 400px; }")
        html_parts.append("        .path-cell { word-break: break-all; }")
        html_parts.append("        .stats-table { margin-bottom: 30px; }") # Style for stats table
        html_parts.append("        .stats-table th { text-align: right; padding-right: 15px; }") 
        html_parts.append("        .stats-table td { text-align: left; }")
        html_parts.append("    </style>")
        html_parts.append("</head>")
        html_parts.append("<body>")

        # --- Add Statistics to HTML ---
        if stats_data and "overall_summary_stats" in stats_data:
            html_parts.append("    <h2>PDF Processing Statistics</h2>")
            html_parts.append("    <table class=\"stats-table\">")
            overall_stats = stats_data["overall_summary_stats"]
            stats_map = {
                "total_pdfs_processed": "Total PDFs Processed:",
                "total_pages_in_all_pdfs": "Total Pages in All PDFs:",
                "total_records_extracted": "Total Records Extracted:",
                "total_images_saved": "Total Images Saved:",
                "total_records_without_images": "Total Records Without Images:",
                "total_failed_image_extractions": "Total Failed Image Extractions:"
            }
            for key, label in stats_map.items():
                if key in overall_stats:
                    html_parts.append("        <tr>")
                    html_parts.append(f"            <th>{label}</th>")
                    html_parts.append(f"            <td>{overall_stats[key]}</td>")
                    html_parts.append("        </tr>")
            html_parts.append("    </table>")

        if stats_data and "pdf_processing_summary" in stats_data:
            html_parts.append("    <h3>Per-PDF Processing Summary</h3>")
            html_parts.append("    <table>") # Use default table style
            html_parts.append("        <thead>")
            html_parts.append("            <tr>")
            html_parts.append("                <th>PDF Name</th>")
            html_parts.append("                <th>Pages Processed</th>")
            html_parts.append("                <th>Successful Images</th>")
            html_parts.append("                <th>Failed Images</th>")
            html_parts.append("            </tr>")
            html_parts.append("        </thead>")
            html_parts.append("        <tbody>")
            for item in stats_data["pdf_processing_summary"]:
                failed_images = item.get('total_pages_processed', 0) - item.get('successful_images', 0)
                html_parts.append("            <tr>")
                html_parts.append(f"                <td>{item.get('pdf_name', 'N/A')}</td>")
                html_parts.append(f"                <td>{item.get('total_pages_processed', 0)}</td>")
                html_parts.append(f"                <td>{item.get('successful_images', 0)}</td>")
                html_parts.append(f"                <td>{failed_images}</td>")
                html_parts.append("            </tr>")
            html_parts.append("        </tbody>")
            html_parts.append("    </table>")
        # ------------------------------

        html_parts.append("    <h2>Organized Data</h2>")
        html_parts.append("    <table>")
        html_parts.append("        <thead>")
        html_parts.append("            <tr>")
        for col_name in base_columns:
            html_parts.append(f"                <th>{col_name}</th>")
        html_parts.append("            </tr>")
        html_parts.append("        </thead>")
        html_parts.append("        <tbody>")

        # Populate table rows
        for index, row in df.iterrows():
            html_parts.append("            <tr>")
            for col_name in base_columns:
                cell_data = row.get(col_name, "")
                if col_name == "Image" and include_actual_images:
                    image_path = row.get("Image Path") # Get the original image path
                    img_base64 = get_image_base64(image_path)
                    if img_base64:
                        html_parts.append(f'                <td class="image-cell"><img src="data:image/png;base64,{img_base64}" alt="Image"></td>')
                    elif image_path:
                        html_parts.append(f'                <td class="image-cell path-cell">Not found: {image_path}</td>')
                    else:
                        html_parts.append('                <td class="image-cell">No image</td>')
                elif col_name == "Image Path" and not include_actual_images:
                    html_parts.append(f'                <td class="path-cell">{cell_data}</td>')
                else:
                    html_parts.append(f"                <td>{cell_data}</td>")
            html_parts.append("            </tr>")

        html_parts.append("        </tbody>")
        html_parts.append("    </table>")
        html_parts.append("</body>")
        html_parts.append("</html>")

        html_content = "\n".join(html_parts) # Join parts with single newlines

        # Write to HTML file
        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Data successfully organized and saved to {html_file_path}")
        if not include_actual_images:
            print("Note: Actual images were NOT embedded in this run due to the 'include_actual_images' flag.")
            print("Image paths (if available) are shown instead.")

    except FileNotFoundError:
        print(f"Error: The file {json_file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {json_file_path}. Please ensure it's valid JSON.")
    except ImportError as e:
        print(f"Error: A required library is missing. {e}. Please install it.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    services_dir = os.path.dirname(current_script_dir)
    project_root = os.path.dirname(services_dir)

    json_input_path = os.path.join(project_root, "output.json")
    # Changed output path to HTML
    html_output_path = os.path.join(project_root, "organized_data_report.html")
    # Define stats file path relative to project root as well for the main execution
    stats_input_path = os.path.join(project_root, "processing_stats.json")

    # Pass the stats_file_path to the function, though the function itself will use a relative path for now
    # For consistency, we could make the function accept it, but it's currently hardcoded.
    # The main block here illustrates where it *would* be if passed.
    organize_data_to_html(json_input_path, html_output_path)
