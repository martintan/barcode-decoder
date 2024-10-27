import os
import sys
from constants import DOC_HEIGHT, DOC_WIDTH
from yolo_localizer import (
    load_yolo_model,
    localize_barcode_in_image,
    setup_yolo_detector,
    train_yolo_detector,
)
import shutil
from pdf_to_jpg import pdf_to_jpg
from typing import Callable, NamedTuple
from dataclasses import dataclass

from doctype1 import (
    generate_training_images as gen_train_type1,
    generate_image as gen_image_type1,
)

from doctype2 import (
    generate_training_images as gen_train_type2,
    generate_image as gen_image_type2,
)

from doctype3 import (
    generate_training_images as gen_train_type3,
    generate_image as gen_image_type3,
)

@dataclass
class DocTypeConfig:
    name: str
    generate_func: Callable
    percentage: float

# Define document types and their configurations
DOC_TYPES = [
    DocTypeConfig(
        name="type1",
        generate_func=lambda **kwargs: gen_train_type1(generate_func=gen_image_type1, **kwargs),
        percentage=0.4
    ),
    DocTypeConfig(
        name="type2",
        generate_func=lambda **kwargs: gen_train_type2(generate_func=gen_image_type2, **kwargs),
        percentage=0.4
    ),
    DocTypeConfig(
        name="type3",
        generate_func=lambda **kwargs: gen_train_type3(generate_func=gen_image_type3, **kwargs),
        percentage=0.2
    ),
]

def print_usage():
    print("Usage: python main.py [mode] [file_path/num_images]")
    print("Modes:")
    print("  generate [num_images] - Generate training images (default: 5000)")
    print("  localize [file_path] - Localize barcode in the provided image")
    print(
        "If no mode is specified, the program will train or use the YOLO model as before."
    )


if __name__ == "__main__":
    yolo_num_training_images = 20
    yolo_model_path = "yolo.pt"

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "generate":
        if len(sys.argv) > 2:
            try:
                yolo_num_training_images = int(sys.argv[2])
            except ValueError:
                print("Invalid number of images. Using default value of 5000.")

        # Validate percentages sum to 1.0
        total_percentage = sum(doc_type.percentage for doc_type in DOC_TYPES)
        if not abs(total_percentage - 1.0) < 0.0001:
            print(f"Error: Document type percentages must sum to 1.0 (current sum: {total_percentage})")
            sys.exit(1)

        start_index = 0
        for doc_type in DOC_TYPES:
            # Calculate number of images for this type
            type_count = int(yolo_num_training_images * doc_type.percentage)
            
            # Generate images for this document type
            doc_type.generate_func(
                num_images=type_count,
                training_folder="training",
                force_generate=(start_index == 0),  # Only force generate for first type
                start_index=start_index
            )
            
            print(f"Generated {type_count} {doc_type.name} images")
            start_index += type_count

        print(f"Training image generation complete. Total images: {yolo_num_training_images}")
        sys.exit(0)

    if os.path.exists(yolo_model_path):
        yolo_model = load_yolo_model(yolo_model_path)

        if mode == "localize":
            if len(sys.argv) < 3:
                print("Please provide a PDF or JPG file path as an argument.")
                sys.exit(1)

            input_file = sys.argv[2]
            file_extension = os.path.splitext(input_file)[1].lower()

            if file_extension == ".pdf":
                jpg_file = pdf_to_jpg(input_file)
            elif file_extension in [".jpg", ".jpeg"]:
                jpg_file = input_file
            else:
                print("Unsupported file format. Please provide a PDF or JPG file.")
                sys.exit(1)

            # Localize barcode in the image using the jpg_file path
            localize_barcode_in_image(yolo_model, jpg_file)
        else:
            print_usage()
            sys.exit(1)
    else:
        if mode != "localize":
            print_usage()
            sys.exit(1)
        else:
            # Validate percentages sum to 1.0
            total_percentage = sum(doc_type.percentage for doc_type in DOC_TYPES)
            if not abs(total_percentage - 1.0) < 0.0001:
                print(f"Error: Document type percentages must sum to 1.0 (current sum: {total_percentage})")
                sys.exit(1)

            start_index = 0
            for doc_type in DOC_TYPES:
                # Calculate number of images for this type
                type_count = int(yolo_num_training_images * doc_type.percentage)
                
                # Generate images for this document type
                doc_type.generate_func(
                    num_images=type_count,
                    training_folder="training",
                    force_generate=(start_index == 0),  # Only force generate for first type
                    start_index=start_index
                )
                
                print(f"Generated {type_count} {doc_type.name} images")
                start_index += type_count

            print(f"Training image generation complete. Total images: {yolo_num_training_images}")

            yolo_model = setup_yolo_detector(DOC_WIDTH, DOC_HEIGHT)
            train_yolo_detector(yolo_model, DOC_WIDTH, DOC_HEIGHT)
            print("YOLO detector training complete and model saved.")

            shutil.rmtree("training")
