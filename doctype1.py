from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import numpy as np
import cv2
import os

from constants import DOC_HEIGHT, DOC_WIDTH
from utils import (
    add_text_to_image,
    generate_random_number,
    generate_random_text,
    generate_random_word,
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


def create_barcode_image(
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


def add_scan_effects(image_path: str, apply_noise: bool = False):
    image = Image.open(image_path)
    image = image.convert("L")

    image_array = np.array(image, dtype=np.int16)

    if apply_noise:
        noisy_image = add_noise(image_array)
    else:
        noisy_image = image_array

    brightness_map, alpha = create_brightness_map(image_array.shape)

    image_array = image_array.astype(float)

    # Blend original image with brightness map using pixel-wise alpha
    blended_image = (1 - alpha) * noisy_image + alpha * (brightness_map * 255)

    # Apply damaged pixels effect
    blended_image = apply_damaged_pixels(blended_image)

    final_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    blur_kernel = np.random.randint(1, 3, size=(2,))
    final_image = cv2.blur(
        final_image, tuple(blur_kernel), borderType=cv2.BORDER_REFLECT
    )

    image = Image.fromarray(final_image, mode="L")

    # Convert back to original mode if it wasn't grayscale
    original_mode = Image.open(image_path).mode
    if original_mode != "L":
        image = image.convert(original_mode)

    image.save(image_path, format="JPEG", quality=95)


def create_brightness_map(shape):
    num_lights = np.random.randint(2, 5)
    light_centers = np.random.rand(num_lights, 2)
    light_intensities = np.random.uniform(0.1, 0.4, num_lights)

    y, x = np.ogrid[: shape[0], : shape[1]]
    brightness_map = np.ones(shape, dtype=float)

    for center, intensity in zip(light_centers, light_intensities):
        dist_from_center = np.sqrt(
            (x / shape[1] - center[0]) ** 2 + (y / shape[0] - center[1]) ** 2
        )
        light_effect = np.exp(-dist_from_center * 1.2) * intensity
        brightness_map += light_effect

    # Normalize brightness map and apply a power function to increase bright areas
    brightness_map = (brightness_map - brightness_map.min()) / (
        brightness_map.max() - brightness_map.min()
    )
    brightness_map = np.power(brightness_map, 0.7)

    # Calculate alpha based on brightness map
    alpha_min, alpha_max = 0.1, 0.9
    alpha = alpha_min + (alpha_max - alpha_min) * brightness_map

    return brightness_map, alpha


def generate_training_images(num_images: int, training_folder: str) -> None:
    images_folder = os.path.join(training_folder, "images")
    labels_folder = os.path.join(training_folder, "labels")
    os.makedirs(labels_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    existing_images = len([f for f in os.listdir(images_folder) if f.endswith(".jpg")])
    images_to_generate = max(0, num_images - existing_images)

    print(
        f"Found {existing_images} existing images. Generating {images_to_generate} new images."
    )

    for i in range(existing_images, num_images):
        image_number = i + 1
        image_filename = f"barcode_{image_number}.png"
        create_barcode_image(training_folder, image_filename, DOC_WIDTH, DOC_HEIGHT)
        print(f"Generated image {image_number}/{num_images}")


def add_noise(image_array):
    noise = np.random.choice([0, 127, 255], size=image_array.shape).astype(np.int16)
    random_mask = np.random.random(image_array.shape) < 0.0025
    noisy_image = image_array.copy()
    noisy_image[random_mask] = noise[random_mask]
    return noisy_image


def apply_damaged_pixels(image_array):
    damaged_pixels = np.random.random(image_array.shape) < 0.01
    white_pixels = np.random.random(image_array.shape) < 0.7
    dark_pixels = np.logical_and(
        ~white_pixels, np.random.random(image_array.shape) < 0.3
    )
    image_array[np.logical_and(damaged_pixels, white_pixels)] = 255
    dark_intensity = np.random.uniform(0.3, 0.7, size=image_array.shape)
    dark_pixel_values = np.clip(image_array * dark_intensity, 100, 192)
    image_array[np.logical_and(damaged_pixels, dark_pixels)] = dark_pixel_values[
        np.logical_and(damaged_pixels, dark_pixels)
    ]

    return image_array


if __name__ == "__main__":
    generate_training_images(5, "training")
    print("Training image generation complete.")
