from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont
import random

from utils import (
    apply_blur_effect,
    add_text_to_image,
    apply_pixel_damage_effect,
    generate_training_images,
    np_to_pil_grayscale,
    pil_to_np_grayscale,
    generate_random_number,
    generate_random_text,
    generate_random_word,
    save_pil_jpeg,
)


def add_text_and_lines(
    draw, doc_width, doc_height, font_small, font_medium, font_large
):
    def draw_horizontal_line(y_position, x_start, x_end):
        draw.line([(x_start, y_position), (x_end, y_position)], fill="black", width=1)

    def add_text_block(texts, start_x, start_y, fonts):
        for i, (text, font) in enumerate(zip(texts, fonts)):
            add_text_to_image(draw, text, (start_x, start_y + i * 12), font)

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

        add_text_block(texts, left_x_start, y_start, fonts)
        draw_horizontal_line(y_start + 36, left_x_start, left_x_end)

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

        add_text_block(texts, right_x_start, y_start, fonts)
        draw_horizontal_line(y_start + 36, right_x_start, right_x_end)

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
    draw_horizontal_line(doc_height - 20, bottom_x_start, bottom_x_end)


def generate_image(
    training_folder: str, filename: str, doc_width: int, doc_height: int
) -> None:
    background = Image.new("RGB", (doc_width, doc_height), color="white")
    draw = ImageDraw.Draw(background)

    font_small = ImageFont.truetype("opensans.ttf", 8)
    font_medium = ImageFont.truetype("opensans.ttf", 10)
    font_large = ImageFont.truetype("opensans.ttf", 48)

    add_text_and_lines(draw, doc_width, doc_height, font_small, font_medium, font_large)

    barcode_data = generate_random_number()
    options = {
        "module_width": 0.12,
        "module_height": 2.4,
        "font_size": 2,
        "text_distance": 1,
        "foreground": "black",
        "center_text": True,
        "quiet_zone": 0,
    }
    barcode = Code128(barcode_data, writer=ImageWriter())
    barcode_img = barcode.render(options, text=barcode_data)
    barcode_position = (20, doc_height - barcode_img.height - 20)
    background.paste(barcode_img, barcode_position)

    # Calculate barcode bounding box
    barcode_x = 20
    barcode_y = doc_height - barcode_img.height - 10
    barcode_width = barcode_img.width
    barcode_height = barcode_img.height - 35

    # Update the save location and file extension
    barcode_path = f"{training_folder}/images/{filename.rsplit('.', 1)[0]}.jpg"
    background.save(barcode_path, format="JPEG", quality=95)

    # Calculate normalized bounding box coordinates
    norm_barcode_x = barcode_x / doc_width
    norm_barcode_y = barcode_y / doc_height
    norm_barcode_width = barcode_width / doc_width
    norm_barcode_height = barcode_height / doc_height

    # Calculate center coordinates and dimensions
    center_x = norm_barcode_x + (norm_barcode_width / 2)
    center_y = norm_barcode_y + (norm_barcode_height / 2)

    # Create label file
    label_name = filename.split("/")[-1].rsplit(".", 1)[0]
    label_path = f"{training_folder}/labels/{label_name}.txt"
    with open(label_path, "w") as f:
        f.write(
            f"0 {center_x:.6f} {center_y:.6f} {norm_barcode_width:.6f} {norm_barcode_height:.6f}"
        )

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
