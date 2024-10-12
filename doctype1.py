from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np
import cv2

from utils import (
    add_text_to_image,
    generate_random_number,
    generate_random_text,
    generate_random_word,
)


def create_barcode_image(data: str, output_path: str) -> None:
    doc_width, doc_height = 1600, 800
    background = Image.new("RGB", (doc_width, doc_height), color="white")
    draw = ImageDraw.Draw(background)

    font_small = ImageFont.truetype("opensans.ttf", 20)
    font_medium = ImageFont.truetype("opensans.ttf", 24)
    font_large = ImageFont.truetype("opensans.ttf", 120)

    def draw_horizontal_line(y_position, x_start, x_end):
        draw.line([(x_start, y_position), (x_end, y_position)], fill="black", width=1)

    def add_text_block(texts, start_x, start_y, fonts):
        for i, (text, font) in enumerate(zip(texts, fonts)):
            add_text_to_image(draw, text, (start_x, start_y + i * 30), font)

    # Left column
    left_x_start, left_x_end = 50, 400

    for i in range(4):
        y_start = 120 + i * 100
        texts = [
            generate_random_text(3),
            generate_random_text(4),
            generate_random_text(5),
        ]
        fonts = [font_small, font_medium, font_small]

        if i == 1:
            texts[0] = "CONSIGNEE"
            texts[1] = generate_random_text(2).upper()
        elif i == 2:
            texts[0] = generate_random_text(2).upper()
            texts[2] = generate_random_text(3).upper()
        elif i == 3:
            texts[2] = f"Contents: {generate_random_text(1)}"

        add_text_block(texts, left_x_start, y_start, fonts)
        draw_horizontal_line(y_start + 90, left_x_start, left_x_end)

    right_x_start, right_x_end = 450, 800

    for i in range(2):
        y_start = 120 + i * 100
        texts = [
            generate_random_text(3),
            generate_random_text(4),
            generate_random_text(5),
        ]
        fonts = [font_small, font_medium, font_small]

        if i == 1:
            texts[1] = generate_random_text(2).upper()

        add_text_block(texts, right_x_start, y_start, fonts)
        draw_horizontal_line(y_start + 90, right_x_start, right_x_end)

    for i in range(3):
        if i < 2:
            text = f"{generate_random_text(1).upper()}: {generate_random_text(1)}"
        else:
            text = generate_random_text(2)
        add_text_to_image(draw, text, (doc_width - 300, 50 + i * 30), font_small)

    large_text = random.choice(
        [str(random.randint(10, 99)), generate_random_word(2).upper()]
    )
    add_text_to_image(draw, large_text, (doc_width - 150, doc_height - 200), font_large)

    barcode_data = generate_random_number()
    options = {
        "module_width": 0.3,
        "module_height": 6,
        "font_size": 5,
        "text_distance": 3,
        "foreground": "black",
        "center_text": True,
    }
    barcode = Code128(barcode_data, writer=ImageWriter())
    barcode_img = barcode.render(options)
    barcode_position = (50, doc_height - barcode_img.height - 50)
    background.paste(barcode_img, barcode_position)

    bottom_x_start = doc_width - 500
    bottom_x_end = doc_width - 50
    bottom_text = generate_random_text(3)
    add_text_to_image(draw, bottom_text, (bottom_x_start, doc_height - 70), font_small)
    draw_horizontal_line(doc_height - 50, bottom_x_start, bottom_x_end)

    background.save(output_path)

    add_scan_effects(output_path)


def add_scan_effects(image_path: str):
    image = Image.open(image_path)

    # Convert to grayscale
    image = image.convert("L")

    # Create subtle noise
    noise = np.random.choice([0, 127, 255], size=(image.height, image.width)).astype(
        np.int16
    )

    # Apply noise to the image
    image_array = np.array(
        image, dtype=np.int16
    )  # Convert to int16 to allow for negative values

    # Create a mask for 2.5% of pixels
    random_mask = np.random.random(image_array.shape) < 0.025

    # Apply noise only to the masked pixels
    noisy_image = image_array.copy()
    noisy_image[random_mask] = noise[random_mask]

    # Create more severe uneven brightness effect
    brightness_map = np.random.normal(loc=1.0, scale=0.2, size=image_array.shape)
    brightness_map = np.clip(
        brightness_map, 0.5, 1.5
    )  # Increase the brightness variation range

    # Apply brightness map to the noisy image
    uneven_image = noisy_image * brightness_map

    # Clip values to ensure they're in the valid range and convert back to uint8
    final_image = np.clip(uneven_image, 0, 255).astype(np.uint8)

    # Add blur to simulate less clear areas
    blur_kernel = np.random.randint(1, 3, size=final_image.shape)
    final_image = cv2.blur(final_image, (3, 3), borderType=cv2.BORDER_REFLECT)

    image = Image.fromarray(final_image, mode="L")

    # Ensure the image is in the same mode as the original
    original_mode = Image.open(image_path).mode
    image = image.convert(original_mode)

    # Save the modified image
    image.save(image_path)
