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


if __name__ == "__main__":
    yolo_num_training_images = 5_000
    yolo_model_path = "yolo.pt"

    if os.path.exists(yolo_model_path):
        yolo_model = load_yolo_model(yolo_model_path)

        # Check if a file path is provided as an argument
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
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
            print("Please provide a PDF or JPG file path as an argument.")
            sys.exit(1)
    else:
        generate_training_images(yolo_num_training_images, training_folder="training")
        print("Training image generation complete.")

        yolo_model = setup_yolo_detector(DOC_WIDTH, DOC_HEIGHT)
        train_yolo_detector(yolo_model, DOC_WIDTH, DOC_HEIGHT)
        print("YOLO detector training complete and model saved.")

        shutil.rmtree("training")
