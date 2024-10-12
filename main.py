import os

from doctype1 import create_barcode_image
from utils import generate_random_number


def generate_training_images(num_images: int, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_images):
        data = generate_random_number()
        output_path = os.path.join(output_folder, f"barcode_{i+1}.png")
        create_barcode_image(data, output_path)
        print(f"Generated image {i+1}/{num_images}")


if __name__ == "__main__":
    num_training_images = 1
    output_folder = "training"
    generate_training_images(num_training_images, output_folder)
    print("Training image generation complete.")
