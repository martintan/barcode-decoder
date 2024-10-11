from barcode import Code128
from barcode.writer import BaseWriter, ImageWriter
from PIL import Image, ImageDraw, ImageFont
import random
import string
import os


def generate_random_number() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def create_barcode_image(data: str, output_path: str) -> None:
    def generate_dark_color():
        return "#{:02x}{:02x}{:02x}".format(
            random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)
        )

    options = {
        "module_width": 0.3,
        "module_height": 6,
        "font_size": 5,
        "text_distance": 3,
        "foreground": generate_dark_color(),
        "center_text": True,
    }

    barcode = Code128(data, writer=ImageWriter())
    doc_width, doc_height = 1600, 800
    background = Image.new("RGB", (doc_width, doc_height), color="white")
    barcode_img = barcode.render(options, text=data)

    vertical_padding = int(min(doc_width, doc_height) * 0.05)
    horizontal_padding = int(min(doc_width, doc_height) * 0.1)
    position = (horizontal_padding, doc_height - barcode_img.height - vertical_padding)
    background.paste(barcode_img, position)
    background.save(output_path)


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
