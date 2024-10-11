from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont
import random
import string
import os


def generate_random_string(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def create_barcode_image(data: str, output_path: str) -> None:
    options = {
        "module_height": 10,
        "font_size": 5,
        "text_distance": 3,
    }

    barcode = Code128(data, writer=ImageWriter())
    doc_width, doc_height = 1600, 800
    background = Image.new("RGB", (doc_width, doc_height), color="white")
    barcode_img = barcode.render(options)

    padding = int(min(doc_width, doc_height) * 0.05)
    position = (padding, doc_height - barcode_img.height - padding)
    background.paste(barcode_img, position)
    background.save(output_path)


def generate_training_images(num_images: int, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_images):
        data = generate_random_string(10)
        output_path = os.path.join(output_folder, f"barcode_{i+1}.png")
        create_barcode_image(data, output_path)
        print(f"Generated image {i+1}/{num_images}")


if __name__ == "__main__":
    num_training_images = 1
    output_folder = "training"
    generate_training_images(num_training_images, output_folder)
    print("Training image generation complete.")
