import math
import os
import random
import shutil
import string
from typing import Any, Callable, Optional
import PIL
import cv2
import numpy as np
import PIL
from barcode import Code128
from barcode.writer import ImageWriter

from constants import DOC_HEIGHT, DOC_WIDTH


def generate_random_number() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def generate_random_text(word_count, max_word_length=10):
    return " ".join(
        generate_random_word(random.randint(3, max_word_length))
        for _ in range(word_count)
    )


def generate_random_word(length):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def add_text_to_image(draw, text, position, font, fill="black"):
    draw.text(position, text, font=font, fill=fill)


def apply_brightness_effect(image: np.ndarray):
    # Generate random light sources
    num_lights = np.random.randint(2, 5)
    centers = np.random.rand(num_lights, 2)
    intensities = np.random.uniform(0.1, 0.4, num_lights)

    # Create brightness map from light sources
    y, x = np.ogrid[: image.shape[0], : image.shape[1]]
    brightness_map = np.ones_like(image, dtype=float)
    # Create a Gaussian-like brightness distribution for each light source
    # For each center point and intensity, calculate the distance from every pixel
    # to that light source and apply exponential decay. The sum creates overlapping
    # light effects that simulate natural lighting variations.
    brightness_map += sum(
        np.exp(
            -np.sqrt(
                (x / image.shape[1] - c[0]) ** 2 + (y / image.shape[0] - c[1]) ** 2
            )
            * 1.2
        )
        * i
        for c, i in zip(centers, intensities)
    )

    # Normalize and adjust brightness map
    brightness_map = (brightness_map - brightness_map.min()) / (
        brightness_map.max() - brightness_map.min()
    )
    brightness_map = np.power(brightness_map, 0.7)

    # Apply brightness effect
    alpha = 0.1 + 0.8 * brightness_map
    return (1 - alpha) * image + alpha * (brightness_map * 255)


def apply_noise_effect(image: np.ndarray):
    mask = np.random.random(image.shape) < 0.0025
    image[mask] = np.random.choice([0, 127, 255], size=mask.sum())
    return image


def apply_pixel_damage_effect(
    image: np.ndarray,
    white_pixel_pct: float = np.random.uniform(0.01, 0.5),
    dark_pixel_pct: float = 0.01,
):
    # Apply white pixels
    image[
        np.random.random(image.shape)
        < np.random.uniform(white_pixel_pct, white_pixel_pct * 50)
    ] = 255

    # Apply dark pixels to damaged areas
    damaged = np.random.random(image.shape) < dark_pixel_pct
    dark = np.logical_and(damaged, np.random.random(image.shape) < 0.3)
    image[dark] = np.clip(image[dark] * np.random.uniform(0.3, 0.7), 100, 192)

    return image


def apply_blur_effect(image: np.ndarray):
    kernel_size = np.random.randint(1, 3, size=(2,))
    image = cv2.blur(image, tuple(kernel_size), borderType=cv2.BORDER_REFLECT)
    return image


# use this to initialize the image for applying scan effects
def pil_to_np_grayscale(image_path: str) -> np.ndarray:
    image = PIL.Image.open(image_path)
    image = image.convert("L")
    return np.array(image, dtype=np.float32)


def np_to_pil_grayscale(image: np.ndarray) -> PIL.Image.Image:
    return PIL.Image.fromarray(image, mode="L")


def save_pil_jpeg(pil_image: PIL.Image.Image, path: str):
    pil_image.save(path, format="JPEG", quality=95)


def generate_signature_scribble(
    draw: PIL.ImageDraw.Draw,
    start_x: int,
    start_y: int,
    width: int,
):
    def generate_bezier_curve(
        points: list[tuple[int, int]], num_steps: int = 50
    ) -> list[tuple[int, int]]:
        curve_points = []
        for t in range(num_steps):
            t = t / (num_steps - 1)
            x = (
                (1 - t) ** 3 * points[0][0]
                + 3 * (1 - t) ** 2 * t * points[1][0]
                + 3 * (1 - t) * t**2 * points[2][0]
                + t**3 * points[3][0]
            )
            y = (
                (1 - t) ** 3 * points[0][1]
                + 3 * (1 - t) ** 2 * t * points[1][1]
                + 3 * (1 - t) * t**2 * points[2][1]
                + t**3 * points[3][1]
            )
            curve_points.append((int(x), int(y)))
        return curve_points

    # Generate multiple strokes for a more natural signature
    num_strokes = random.randint(2, 4)

    for stroke in range(num_strokes):
        # Increased vertical variation
        x_offset = random.randint(-15, 15)
        y_offset = random.randint(-20, 20)  # Increased range

        p0 = (start_x + x_offset, start_y + y_offset)
        p1 = (
            start_x + width // 3 + random.randint(-30, 30),
            start_y + random.randint(-25, 25),
        )  # More vertical variation
        p2 = (
            start_x + 2 * width // 3 + random.randint(-30, 30),
            start_y + random.randint(-25, 25),
        )  # More vertical variation
        p3 = (
            start_x + width + random.randint(-15, 15),
            start_y + random.randint(-20, 20),
        )  # More vertical variation

        # Generate and draw the curve
        curve_points = generate_bezier_curve([p0, p1, p2, p3])

        # More variation in line thickness
        thickness = random.randint(1, 3)  # Increased max thickness
        draw.line(curve_points, fill="black", width=thickness)

        # Add more varied decorative strokes
        for _ in range(random.randint(1, 3)):  # Multiple decorative strokes
            if random.random() < 0.8:  # Increased probability
                x = start_x + random.randint(0, width)
                y = start_y + random.randint(-20, 20)  # Increased vertical range
                length = random.randint(15, 40)  # Longer strokes
                angle = random.randint(-60, 60)  # More extreme angles
                end_x = x + length * math.cos(math.radians(angle))
                end_y = y + length * math.sin(math.radians(angle))
                draw.line(
                    [(x, y), (int(end_x), int(end_y))],
                    fill="black",
                    width=random.randint(1, 2),
                )


def generate_training_images(
    generate_func: Callable[[str, str, int, int], None],
    num_images: int,
    training_folder: str,
    force_generate: bool = False,
    start_index: int = 0,
) -> None:
    images_folder = os.path.join(training_folder, "images")
    labels_folder = os.path.join(training_folder, "labels")
    os.makedirs(labels_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    if force_generate:
        # Only clear directories if force_generate and start_index is 0
        if start_index == 0:
            shutil.rmtree(training_folder)
            os.makedirs(labels_folder, exist_ok=True)
            os.makedirs(images_folder, exist_ok=True)
        images_to_generate = num_images
        print(f"Generating {num_images} images starting at index {start_index}.")
    else:
        images_to_generate = num_images
        print(f"Generating {num_images} images starting at index {start_index}.")

    for i in range(images_to_generate):
        image_number = start_index + i + 1
        image_filename = f"barcode_{image_number}.png"
        generate_func(training_folder, image_filename, DOC_WIDTH, DOC_HEIGHT)
        print(f"Generated image {i+1}/{images_to_generate}")


def load_all_fonts() -> tuple:
    font_small = PIL.ImageFont.truetype("opensans.ttf", 8)
    font_medium = PIL.ImageFont.truetype("opensans.ttf", 10)
    font_large = PIL.ImageFont.truetype("opensans_bold.ttf", 30)
    font_extra_large = PIL.ImageFont.truetype("opensans_bold.ttf", 48)
    return font_small, font_medium, font_large, font_extra_large


def draw_horizontal_line(
    draw: PIL.ImageDraw.Draw,
    start_y: int,
    start_x: int,
    end_x: int,
) -> None:
    draw.line([(start_x, start_y), (end_x, start_y)], fill="black", width=1)


def draw_vertical_line(
    draw: PIL.ImageDraw.Draw,
    start_x: int,
    start_y: int,
    end_y: int,
) -> None:
    draw.line([(start_x, start_y), (start_x, end_y)], fill="black", width=1)


def add_text_block(
    draw: PIL.ImageDraw.Draw,
    texts: list[str],
    start_x: int,
    start_y: int,
    fonts: list[PIL.ImageFont.FreeTypeFont],
) -> None:
    for i, (text, font) in enumerate(zip(texts, fonts)):
        add_text_to_image(draw, text, (start_x, start_y + i * 12), font)


def add_barcode(
    background: PIL.Image.Image,
    doc_width: int,
    doc_height: int,
    position: Optional[tuple[int, int]] = None,
) -> tuple[PIL.Image.Image, tuple[int, int, int, int]]:
    """
    Adds a barcode to the image and returns the barcode dimensions.

    Args:
        background: PIL Image to add barcode to
        doc_width: Width of the document
        doc_height: Height of the document
        position: Optional (x,y) tuple for barcode position. If None, places at bottom left.

    Returns:
        Tuple of (modified image, (barcode_x, barcode_y, barcode_width, barcode_height))
    """
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

    barcode_x = 0
    barcode_y = 0

    # Set barcode position
    if position:
        barcode_x, barcode_y = position
    else:
        barcode_x = 20
        barcode_y = doc_height - barcode_img.height - 20

    background.paste(barcode_img, (barcode_x, barcode_y))

    # Calculate dimensions for bounding box
    barcode_width = barcode_img.width
    barcode_height = barcode_img.height - 35

    return background, (barcode_x, barcode_y, barcode_width, barcode_height)


def create_yolo_label(
    training_folder: str,
    filename: str,
    doc_width: int,
    doc_height: int,
    barcode_dims: tuple[int, int, int, int],
) -> None:
    """
    Creates a YOLO format label file for the barcode location.

    Args:
        training_folder: Folder path for saving files
        filename: Name of the file
        doc_width: Width of the document
        doc_height: Height of the document
        barcode_dims: Tuple of (x, y, width, height) of the barcode
    """
    barcode_x, barcode_y, barcode_width, barcode_height = barcode_dims

    # Calculate normalized coordinates
    norm_barcode_x = barcode_x / doc_width
    norm_barcode_y = barcode_y / doc_height
    norm_barcode_width = barcode_width / doc_width
    norm_barcode_height = barcode_height / doc_height

    # Calculate center coordinates
    center_x = norm_barcode_x + (norm_barcode_width / 2)
    center_y = norm_barcode_y + (norm_barcode_height / 2)

    # Create label file
    label_name = filename.split("/")[-1].rsplit(".", 1)[0]
    label_path = f"{training_folder}/labels/{label_name}.txt"
    with open(label_path, "w") as f:
        f.write(
            f"0 {center_x:.6f} {center_y:.6f} {norm_barcode_width:.6f} {norm_barcode_height:.6f}"
        )
