from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont
import random

from utils import (
    add_text_block,
    apply_blur_effect,
    add_text_to_image,
    apply_pixel_damage_effect,
    draw_horizontal_line,
    generate_training_images,
    load_all_fonts,
    np_to_pil_grayscale,
    pil_to_np_grayscale,
    generate_random_number,
    generate_random_text,
    generate_random_word,
    save_pil_jpeg,
    add_barcode,
    create_yolo_label,
)


def add_text_and_lines(draw, doc_width, doc_height):
    font_small, font_medium, font_large = load_all_fonts()

    left_x_start, left_x_end = 20, 160
    right_x_start, right_x_end = 180, 320

    # Left side text blocks
    for i in range(4):
        y_start = 48 + i * 40
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

        add_text_block(draw, texts, left_x_start, y_start, fonts)
        draw_horizontal_line(draw, y_start + 36, left_x_start, left_x_end)

    # Right side text blocks
    for i in range(2):
        y_start = 48 + i * 40
        texts = [
            generate_random_text(3),
            generate_random_text(4),
            generate_random_text(5),
        ]
        fonts = [font_small, font_medium, font_small]

        if i == 1:
            texts[1] = generate_random_text(2).upper()

        add_text_block(draw, texts, right_x_start, y_start, fonts)
        draw_horizontal_line(draw, y_start + 36, right_x_start, right_x_end)

    # Top right text
    for i in range(3):
        if i < 2:
            text = f"{generate_random_text(1).upper()}: {generate_random_text(1)}"
        else:
            text = generate_random_text(2)
        add_text_to_image(draw, text, (doc_width - 120, 20 + i * 12), font_small)

    # Large text
    large_text = random.choice(
        [str(random.randint(10, 99)), generate_random_word(2).upper()]
    )
    add_text_to_image(draw, large_text, (doc_width - 60, doc_height - 80), font_large)

    # Bottom text
    bottom_x_start = doc_width - 200
    bottom_x_end = doc_width - 20
    bottom_text = generate_random_text(3)
    add_text_to_image(draw, bottom_text, (bottom_x_start, doc_height - 28), font_small)
    draw_horizontal_line(draw, doc_height - 20, bottom_x_start, bottom_x_end)


def generate_image(
    training_folder: str,
    filename: str,
    doc_width: int,
    doc_height: int,
) -> None:
    background = Image.new("RGB", (doc_width, doc_height), color="white")
    draw = ImageDraw.Draw(background)
    add_text_and_lines(draw, doc_width, doc_height)
    background, barcode_dims = add_barcode(background, doc_width, doc_height)
    create_yolo_label(training_folder, filename, doc_width, doc_height, barcode_dims)
    barcode_path = f"{training_folder}/images/{filename.rsplit('.', 1)[0]}.jpg"
    background.save(barcode_path, format="JPEG", quality=95)
    add_scan_effects(barcode_path)


def add_scan_effects(image_path: str):
    image = pil_to_np_grayscale(image_path)
    image = apply_pixel_damage_effect(image)
    image = apply_blur_effect(image)
    image = np_to_pil_grayscale(image)
    save_pil_jpeg(image, image_path)


if __name__ == "__main__":
    generate_training_images(
        generate_func=generate_image,
        num_images=5,
        training_folder="training",
        force_generate=True,
        start_index=0,
    )
    print("Training image generation complete.")
