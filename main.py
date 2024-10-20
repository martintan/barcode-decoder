import os
from constants import DOC_HEIGHT, DOC_WIDTH
from doctype1 import generate_training_images
from yolo_localizer import (
    load_yolo_model,
    localize_barcode_in_image,
    setup_yolo_detector,
    train_yolo_detector,
)
import shutil


if __name__ == "__main__":

    yolo_num_training_images = 5_000
    yolo_model_path = "yolo.pt"

    if os.path.exists(yolo_model_path):
        yolo_model = load_yolo_model(yolo_model_path)

        # Localize barcode in the sample image
        sample_image_path = "./samples/doctype1.jpg"
        localize_barcode_in_image(yolo_model, sample_image_path)
    else:
        generate_training_images(yolo_num_training_images, training_folder="training")
        print("Training image generation complete.")

        yolo_model = setup_yolo_detector(DOC_WIDTH, DOC_HEIGHT)
        train_yolo_detector(yolo_model, DOC_WIDTH, DOC_HEIGHT)
        print("YOLO detector training complete and model saved.")

        shutil.rmtree("training")
