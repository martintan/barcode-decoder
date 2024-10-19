import os
from constants import DOC_HEIGHT, DOC_WIDTH
from doctype1 import create_barcode_image
from yolo_localizer import (
    load_yolo_model,
    localize_barcode_in_image,
    setup_yolo_detector,
    train_yolo_detector,
)


def generate_training_images(num_images: int, output_folder: str) -> None:
    if os.path.exists(output_folder):
        for root, dirs, files in os.walk(output_folder, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_images):
        create_barcode_image(f"barcode_{i+1}.png", DOC_WIDTH, DOC_HEIGHT)
        print(f"Generated image {i+1}/{num_images}")


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
