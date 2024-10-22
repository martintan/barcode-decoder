import os
import sys
from constants import DOC_HEIGHT, DOC_WIDTH
from doctype1 import generate_training_images
from yolo_localizer import (
    load_yolo_model,
    localize_barcode_in_image,
    setup_yolo_detector,
    train_yolo_detector,
)
import shutil
from pdf_to_jpg import pdf_to_jpg


def print_usage():
    print("Usage: python main.py [mode] [file_path/num_images]")
    print("Modes:")
    print("  generate [num_images] - Generate training images (default: 5000)")
    print("  localize [file_path] - Localize barcode in the provided image")
    print(
        "If no mode is specified, the program will train or use the YOLO model as before."
    )


if __name__ == "__main__":
    yolo_num_training_images = 5_000
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
        generate_training_images(
            yolo_num_training_images, training_folder="training", force_generate=True
        )
        print(
            f"Training image generation complete. Generated {yolo_num_training_images} images."
        )
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
            generate_training_images(
                yolo_num_training_images, training_folder="training"
            )
            print("Training image generation complete.")

            yolo_model = setup_yolo_detector(DOC_WIDTH, DOC_HEIGHT)
            train_yolo_detector(yolo_model, DOC_WIDTH, DOC_HEIGHT)
            print("YOLO detector training complete and model saved.")

            shutil.rmtree("training")
        else:
            print("YOLO model not found. Please train the model first.")
            sys.exit(1)
