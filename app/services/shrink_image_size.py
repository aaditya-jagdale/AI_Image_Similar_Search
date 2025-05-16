from PIL import Image
import os

def shrink_images(source_dir="images", target_dir="compressed_images", quality=85, max_size=(1920, 1080)):
    """
    Shrinks images from the source directory and saves them to the target directory.

    :param source_dir: Directory containing original images.
    :param target_dir: Directory to save compressed images.
    :param quality: JPEG quality (0-100). Higher is better quality, larger file.
    :param max_size: Tuple (width, height) for maximum dimensions. Images larger than this will be resized.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)

            try:
                with Image.open(source_path) as img:
                    # Preserve original format, but convert RGBA PNGs to RGB for JPG compatibility
                    if img.mode == 'RGBA' and filename.lower().endswith('.png'):
                        img = img.convert('RGB')
                    
                    # Resize if image is larger than max_size, maintaining aspect ratio
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)

                    # Save with optimization and specified quality for JPEGs
                    if filename.lower().endswith(('.jpg', '.jpeg')):
                        img.save(target_path, "JPEG", quality=quality, optimize=True)
                    elif filename.lower().endswith('.png'):
                         # For PNGs, use optimize and compress_level
                         # compress_level (0-9), 9 is max compression. PIL default is 6.
                         # Using optimize=True can take longer but might produce smaller files.
                        img.save(target_path, "PNG", optimize=True, compress_level=9)
                    else:
                        # For other formats, just save (some might not support quality/optimize)
                        img.save(target_path)
                    
                    original_size = os.path.getsize(source_path) / (1024 * 1024) # MB
                    compressed_size = os.path.getsize(target_path) / (1024 * 1024) # MB
                    print(f"Processed {filename}: {original_size:.2f}MB -> {compressed_size:.2f}MB")

            except Exception as e:
                print(f"Could not process {filename}: {e}")

if __name__ == "__main__":
    # Example usage:
    # shrink_images() # Uses default source 'images' and target 'compressed_images'
    
    # Or specify directories:
    shrink_images(source_dir="images", target_dir="compressed_images")
    print("Image compression complete.")
