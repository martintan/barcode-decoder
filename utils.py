import math
import os
import random
import shutil
import string
from typing import Any, Callable
import PIL
import cv2
import numpy as np
import PIL

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
