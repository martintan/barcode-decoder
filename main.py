import os
from constants import DOC_HEIGHT, DOC_WIDTH
from doctype1 import generate_training_images
from yolo_localizer import (
    load_yolo_model,
    localize_barcode_in_image,
    setup_yolo_detector,
    train_yolo_detector,
)


if __name__ == "__main__":
    num_training_images = 5
    output_folder = "training"
    model_path = "yolo.pt"

    if os.path.exists(model_path):
        yolo_model = load_yolo_model(model_path)

        # Localize barcode in the sample image
        sample_image_path = "./samples/doctype1.jpg"
        localize_barcode_in_image(yolo_model, sample_image_path)
    else:
        generate_training_images(num_training_images, output_folder)
        print("Training image generation complete.")

        yolo_model = setup_yolo_detector(DOC_WIDTH, DOC_HEIGHT)
        train_yolo_detector(yolo_model, DOC_WIDTH, DOC_HEIGHT)
        print("YOLO detector training complete and model saved.")
