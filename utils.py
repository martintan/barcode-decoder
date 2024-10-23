import random
import string

import numpy as np


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
