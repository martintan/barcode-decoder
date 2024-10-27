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
from PIL.ImageDraw import ImageDraw
from PIL.Image import Image, fromarray
from PIL.ImageFont import truetype
from barcode import Code128
from barcode.writer import ImageWriter

from constants import DOC_HEIGHT, DOC_WIDTH

DEBUG_DRAW_BOXES = True


def generate_random_number() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(12))


def generate_random_text(word_count, max_word_length=10):
    return " ".join(generate_random_word(random.randint(3, max_word_length)) for _ in range(word_count))


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
        np.exp(-np.sqrt((x / image.shape[1] - c[0]) ** 2 + (y / image.shape[0] - c[1]) ** 2) * 1.2) * i
        for c, i in zip(centers, intensities)
    )

    # Normalize and adjust brightness map
    brightness_map = (brightness_map - brightness_map.min()) / (brightness_map.max() - brightness_map.min())
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
    white_pixel_pct: float = np.random.uniform(0.01, 0.03),
    dark_pixel_pct: float = 0.01,
):
    # Apply white pixels
    image[np.random.random(image.shape) < np.random.uniform(white_pixel_pct, white_pixel_pct * 35)] = 255

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
def pil_to_np_grayscale(image: Image) -> np.ndarray:
    image = image.convert("L")
    return np.array(image, dtype=np.float32)


def np_to_pil(image: np.ndarray, colored: bool = False) -> Image:
    image = np.clip(image, 0, 255).astype(np.uint8)
    return fromarray(image, mode="RGB" if colored else "L")


def save_pil_jpeg(pil_image: Image, path: str):
    pil_image.save(path, format="JPEG", quality=95)


def generate_signature_scribble(
    draw: ImageDraw,
    start_x: int,
    start_y: int,
    width: int,
):
    def generate_bezier_curve(points: list[tuple[int, int]], num_steps: int = 50) -> list[tuple[int, int]]:
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
    font_small = truetype("opensans.ttf", 8)
    font_medium = truetype("opensans.ttf", 10)
    font_large = truetype("opensans_bold.ttf", 30)
    font_extra_large = truetype("opensans_bold.ttf", 48)
    return font_small, font_medium, font_large, font_extra_large


def draw_horizontal_line(
    draw: ImageDraw,
    start_y: int,
    start_x: int,
    end_x: int,
) -> None:
    draw.line([(start_x, start_y), (end_x, start_y)], fill="black", width=1)


def draw_vertical_line(
    draw: ImageDraw,
    start_x: int,
    start_y: int,
    end_y: int,
) -> None:
    draw.line([(start_x, start_y), (start_x, end_y)], fill="black", width=1)


def add_text_block(
    draw: ImageDraw,
    texts: list[str],
    start_x: int,
    start_y: int,
    fonts: list[PIL.ImageFont.FreeTypeFont],
) -> None:
    for i, (text, font) in enumerate(zip(texts, fonts)):
        add_text_to_image(draw, text, (start_x, start_y + i * 12), font)


def add_barcode(
    background: Image,
    doc_width: int,
    doc_height: int,
    position: Optional[tuple[int, int]] = None,
) -> tuple[Image, tuple[int, int, int, int]]:
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
    barcode_y += 10

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
        f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_barcode_width:.6f} {norm_barcode_height:.6f}")


def apply_perspective_warp(
    image: np.ndarray,
    warp_intensity: float = 0.1,
    barcode_coords: Optional[tuple[float, float, float, float]] = None,
) -> tuple[np.ndarray, Optional[tuple[float, float, float, float]]]:
    """
    Returns: Tuple of (warped_image, new_barcode_coords or None if no warp applied)
    """
    # 50% chance to skip warping
    if random.random() < 0.5:
        return image, barcode_coords

    height, width = image.shape[:2]
    max_offset = int(width * warp_intensity)
    top_left_offset = random.randint(0, max_offset)
    top_right_offset = random.randint(0, max_offset)

    # Define source and destination points as before
    src_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dst_points = np.float32(
        [
            [0+top_left_offset, 0+top_left_offset],
            [width-1-top_right_offset, 0+top_right_offset],
            [width-1, height-1],
            [0, height-1]
        ]
    )

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Transform barcode coordinates if provided
    new_coords = None
    if barcode_coords is not None:
        x, y, w, h = barcode_coords
        # Convert to corner points
        corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        # Transform corners
        transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)

        # Get new bounding box from transformed corners
        min_x = np.min(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_x = np.max(transformed_corners[:, 0])
        max_y = np.max(transformed_corners[:, 1])

        new_coords = (min_x, min_y, max_x - min_x, max_y - min_y)

    return warped_image, new_coords


def apply_fold_warp(
    image: np.ndarray,
    warp_intensity: float = 0.05,
    barcode_coords: Optional[tuple[float, float, float, float]] = None,
) -> tuple[np.ndarray, Optional[tuple[float, float, float, float]]]:
    """
    Returns: Tuple of (warped_image, new_barcode_coords or None if no warp applied)
    """
    if random.random() < 0.5:
        return image, barcode_coords

    height, width = image.shape[:2]

    # Define the diagonal fold line
    fold_y = int(random.uniform(0.3, 0.7) * height)
    fold_angle = random.uniform(-10, 10)  # Reduced angle range from ±15 to ±10

    # Create mask for separating top and bottom portions
    mask = np.zeros_like(image, dtype=np.float32)
    for x in range(width):
        y_offset = int(np.tan(np.radians(fold_angle)) * (x - width / 2))
        y_fold = min(max(0, fold_y + y_offset), height - 1)
        mask[:y_fold, x] = 1

    # Calculate warping parameters with reduced displacement
    max_displacement = int(width * warp_intensity * 0.5)  # Reduced multiplier

    # Find the fold line intersections
    left_y = fold_y + int(np.tan(np.radians(fold_angle)) * (-width / 2))
    right_y = fold_y + int(np.tan(np.radians(fold_angle)) * (width / 2))

    # Source points
    src_points = np.float32([[0, 0], [width - 1, 0], [width - 1, right_y], [0, left_y]])

    # Calculate smaller random displacements
    top_left_offset = random.randint(-max_displacement, max_displacement)
    top_right_offset = random.randint(-max_displacement, max_displacement)
    vertical_offset = int(max_displacement * 0.7)  # Reduced vertical displacement

    # Destination points with reduced warping
    dst_points = np.float32(
        [
            [top_left_offset, vertical_offset],
            [width - 1 + top_right_offset, vertical_offset],
            [width - 1, right_y],
            [0, left_y],
        ]
    )

    # Apply perspective transform
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    # Blend with smoother transition
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    result = image * (1 - mask) + warped * mask

    # Add subtler shadow effect
    shadow_width = 20  # Reduced from 30
    shadow_mask = np.zeros_like(mask)

    for x in range(width):
        y_offset = int(np.tan(np.radians(fold_angle)) * (x - width / 2))
        y_fold = min(max(0, fold_y + y_offset), height - 1)

        for y in range(max(0, y_fold - shadow_width), min(height, y_fold + shadow_width)):
            distance = abs(y - y_fold)
            shadow_intensity = (1 - distance / shadow_width) * 0.25  # Reduced from 0.4
            shadow_mask[y, x] = shadow_intensity

    # Apply shadow
    result = result * (1 - shadow_mask)

    # Transform barcode coordinates if provided
    new_coords = None
    if barcode_coords is not None:
        x, y, w, h = barcode_coords
        corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])

        # Only transform points above the fold line
        transformed_corners = corners.copy()
        for i, corner in enumerate(corners):
            if corner[1] < fold_y:  # Point is above fold
                # Apply similar transformation as the image warp
                offset_x = top_left_offset if corner[0] < width / 2 else top_right_offset
                transformed_corners[i] = [
                    corner[0] + offset_x,
                    corner[1] + vertical_offset,
                ]

        # Get new bounding box
        min_x = np.min(transformed_corners[:, 0])
        min_y = np.min(transformed_corners[:, 1])
        max_x = np.max(transformed_corners[:, 0])
        max_y = np.max(transformed_corners[:, 1])

        new_coords = (min_x, min_y, max_x - min_x, max_y - min_y)

    return result.astype(np.uint8), new_coords


def debug_draw_boxes(image: Image, barcode_dims: tuple[int, int, int, int]):
    if not DEBUG_DRAW_BOXES:
        return

    draw = PIL.ImageDraw.Draw(image)
    draw.rectangle(
        [
            (barcode_dims[0], barcode_dims[1]),
            (barcode_dims[0] + barcode_dims[2], barcode_dims[1] + barcode_dims[3]),
        ],
        outline="red",
        width=1,
    )
