import os
from doctype1 import create_barcode_image
from yolo_detector import setup_yolo_detector, train_yolo_detector

DOC_WIDTH = 640
DOC_HEIGHT = 320


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
    generate_training_images(num_training_images, output_folder)
    print("Training image generation complete.")

    yolo_model = setup_yolo_detector(DOC_WIDTH, DOC_HEIGHT)
    train_yolo_detector(yolo_model)
    print("YOLO detector training complete.")
