import sys
import os
from pdf2image import convert_from_path


def pdf_to_jpg(pdf_path: str):
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path)

    # Create samples directory if it doesn't exist
    os.makedirs("samples", exist_ok=True)

    # Get the original filename without extension
    original_filename = os.path.splitext(os.path.basename(pdf_path))[0]

    # Save each image as JPG
    converted_files = []
    for i, image in enumerate(images):
        output_path = f"samples/{original_filename}_{i+1}.jpg"
        image.save(output_path, "JPEG")
        converted_files.append(output_path)

    print(f"Converted {len(images)} pages to JPG images in the 'samples' directory.")

    # Return the path of the first converted image
    return converted_files[0] if converted_files else None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_to_jpg.py <pdf_file>")
        sys.exit(1)

    pdf_file = sys.argv[1]

    pdf_to_jpg(pdf_file)
