from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime, timedelta
import os
import shutil
import math

from constants import DOC_HEIGHT, DOC_WIDTH
from doctype1 import add_scan_effects
from utils import (
    add_text_to_image,
    generate_random_number,
    generate_random_text,
)


def add_text_and_lines_lbc(
    draw, doc_width, doc_height, font_small, font_medium, font_large, font_extra_large
):
    def draw_horizontal_line(y_position, x_start, x_end):
        draw.line([(x_start, y_position), (x_end, y_position)], fill="black", width=1)

    def draw_vertical_line(x_position, y_start, y_end):
        draw.line([(x_position, y_start), (x_position, y_end)], fill="black", width=1)

    # Add LBC logo placeholder (left corner)
    draw.rectangle([(20, 20), (100, 100)], outline="black")

    # Add large bold tracking numbers
    tracking_number = generate_random_text(1, max_word_length=3)
    tracking_x = (doc_width - 150) - 250 + random.randint(-20, 20)
    tracking_y = 80 + random.randint(-10, 10)
    add_text_to_image(draw, tracking_number, (tracking_x, tracking_y), font_extra_large)

    # Add second tracking number
    tracking_number2 = generate_random_text(1, max_word_length=3)
    tracking_x2 = tracking_x + 200  # Offset to the right
    tracking_y2 = 80 + random.randint(-100, 100)  # Independent random vertical position
    add_text_to_image(draw, tracking_number2, (tracking_x2, tracking_y2), font_large)

    # Add third tracking number
    tracking_number3 = generate_random_text(1, max_word_length=3)
    tracking_x3 = tracking_x + 200  # Offset to the right of second number
    tracking_y3 = 80 + random.randint(-100, 100)  # Independent random vertical position
    add_text_to_image(draw, tracking_number3, (tracking_x3, tracking_y3), font_large)

    # Add return code legend (right corner)
    legend_start_x = doc_width - 200
    legend_start_y = 40
    codes = [
        ("Return Code Legend", "RC"),
        ("Delivered", "DEL"),
        ("No ID", "NID"),
        ("Refuse to Accept", "RTA"),
        ("Released to Representative", "REP"),
        ("Consignee Out", "OUT"),
        ("Moved to Unknown Address", "MOV"),
        ("Incorrect Address", "INC"),
        ("Holiday", "HOL"),
        ("Force of Nature", "FOR"),
    ]

    # Draw legend table
    row_height = 15  # Reduced from 20
    table_height = row_height * len(codes)

    # Draw outer table border
    draw.rectangle(
        [
            (legend_start_x - 10, legend_start_y - 5),
            (doc_width - 20, legend_start_y + table_height),
        ],
        outline="black",
    )

    # Draw vertical line separating description and code
    code_column_x = doc_width - 60
    draw.line(
        [
            (code_column_x - 10, legend_start_y - 5),
            (code_column_x - 10, legend_start_y + table_height),
        ],
        fill="black",
        width=1,
    )

    # Add legend entries with horizontal lines between rows
    for i, (text, code) in enumerate(codes):
        y = legend_start_y + (i * row_height)
        add_text_to_image(draw, text, (legend_start_x, y), font_small)
        add_text_to_image(draw, code, (code_column_x, y), font_small)

        # Draw horizontal line after each row (except the last one)
        if i < len(codes) - 1:
            draw_horizontal_line(y + row_height, legend_start_x - 10, doc_width - 20)

    # Add main content
    main_y = 120

    # Merchant details section
    field_spacing = 20  # Space between fields

    # Dispatch date and pickup date
    date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime(
        "%m/%d/%y"
    )
    add_text_to_image(draw, f"Dispatch Date:", (20, main_y), font_small)
    draw_horizontal_line(main_y + 15, 100, 300)  # Line after Dispatch Date
    add_text_to_image(draw, f"PICK UP DATE {date}", (120, main_y), font_medium)

    # Merchant details with lines
    merchant_y = main_y + 30
    add_text_to_image(
        draw,
        f"Merchant Ref. No:",
        (20, merchant_y),
        font_small,
    )
    draw_horizontal_line(merchant_y + 15, 120, 400)  # Line after Merchant Ref
    add_text_to_image(
        draw, generate_random_text(8), (120, merchant_y), font_medium
    )  # Add merchant ref

    add_text_to_image(
        draw,
        f"Merchant Name:",
        (20, merchant_y + field_spacing),
        font_small,
    )
    draw_horizontal_line(
        merchant_y + field_spacing + 15, 120, 400
    )  # Line after Merchant Name
    add_text_to_image(
        draw, generate_random_text(12), (120, merchant_y + field_spacing), font_medium
    )  # Add merchant name

    # Consignee details with lines
    consignee_y = merchant_y + field_spacing * 2
    add_text_to_image(
        draw,
        f"Consignee:",
        (20, consignee_y),
        font_small,
    )
    draw_horizontal_line(consignee_y + 15, 100, 400)  # Line after Consignee
    add_text_to_image(
        draw, generate_random_text(10), (100, consignee_y), font_medium
    )  # Add consignee name

    add_text_to_image(
        draw,
        f"Address:",
        (20, consignee_y + field_spacing),
        font_small,
    )
    draw_horizontal_line(
        consignee_y + field_spacing + 15, 100, 400
    )  # Line after Address
    add_text_to_image(
        draw, generate_random_text(10), (100, consignee_y + field_spacing), font_medium
    )  # Add address

    # Add delivery attempt table
    table_y = main_y + 110
    table_headers = [
        "Del. Attempt Date",
        "Del. Status",
        "Type of ID",
        "ID no.",
        "Remarks",
        "Messenger Name",
        "Time of Del.",
    ]

    # Calculate table dimensions
    row_height = 15  # Same as legend table
    table_width = doc_width - 40  # 20px margin on each side
    column_width = table_width // len(table_headers)

    # Draw outer table border
    draw.rectangle(
        [(20, table_y), (doc_width - 20, table_y + (row_height * 4))],  # 4 rows total
        outline="black",
    )

    # Draw header row and vertical lines
    current_x = 20
    for header in table_headers:
        # Draw vertical line (except for first column)
        if current_x > 20:
            draw_vertical_line(current_x, table_y, table_y + (row_height * 4))

        # Add header text
        add_text_to_image(draw, header, (current_x + 5, table_y + 2), font_small)
        current_x += column_width

    # Draw horizontal lines between rows
    for i in range(1, 4):  # 3 data rows
        y = table_y + (row_height * i)
        draw_horizontal_line(y, 20, doc_width - 20)

    # Add signature lines at bottom
    sig_y = table_y + 70

    # First signature
    draw_horizontal_line(sig_y, 20, 200)
    generate_signature_scribble(draw, 20, sig_y, 200)
    random_name1 = generate_random_text(10)
    add_text_to_image(draw, "Signature over printed name", (20, sig_y), font_small)
    add_text_to_image(draw, "(First Name / Last Name)", (20, sig_y + 10), font_small)

    # Second signature
    draw_horizontal_line(sig_y, 250, 350)
    generate_signature_scribble(draw, 250, sig_y, 100)
    add_text_to_image(draw, "Relation to Consignee", (250, sig_y), font_small)

    # Third signature
    draw_horizontal_line(sig_y, 400, 580)
    generate_signature_scribble(draw, 400, sig_y, 180)
    date = datetime.now().strftime("%m/%d/%y")
    add_text_to_image(draw, "Date", (400, sig_y), font_small)


def create_lbc_document(
    training_folder: str, filename: str, doc_width: int, doc_height: int
) -> None:
    # Reuse most of create_barcode_image but modify barcode position
    background = Image.new("RGB", (doc_width, doc_height), color="white")
    draw = ImageDraw.Draw(background)

    font_small = ImageFont.truetype("opensans.ttf", 8)
    font_medium = ImageFont.truetype("opensans.ttf", 10)
    font_large = ImageFont.truetype("opensans_bold.ttf", 30)
    font_extra_large = ImageFont.truetype("opensans_bold.ttf", 48)

    # Generate and add barcode at top center
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

    # Center the barcode at the top
    barcode_x = (doc_width - barcode_img.width) // 2
    barcode_y = 20
    background.paste(barcode_img, (barcode_x, barcode_y))

    add_text_and_lines_lbc(
        draw,
        doc_width,
        doc_height,
        font_small,
        font_medium,
        font_large,
        font_extra_large,
    )

    # Save and process image (reuse existing code)
    barcode_path = f"{training_folder}/images/{filename.rsplit('.', 1)[0]}.jpg"
    background.save(barcode_path, format="JPEG", quality=95)

    # Calculate normalized bounding box coordinates for YOLO
    barcode_width = barcode_img.width
    barcode_height = barcode_img.height - 35

    norm_barcode_x = barcode_x / doc_width
    norm_barcode_y = barcode_y / doc_height
    norm_barcode_width = barcode_width / doc_width
    norm_barcode_height = barcode_height / doc_height

    center_x = norm_barcode_x + (norm_barcode_width / 2)
    center_y = norm_barcode_y + (norm_barcode_height / 2)

    # Create label file
    label_name = filename.split("/")[-1].rsplit(".", 1)[0]
    label_path = f"{training_folder}/labels/{label_name}.txt"
    with open(label_path, "w") as f:
        f.write(
            f"0 {center_x:.6f} {center_y:.6f} {norm_barcode_width:.6f} {norm_barcode_height:.6f}"
        )

    add_scan_effects(barcode_path, apply_brightness=False, apply_noise=False)


def generate_signature_scribble(draw, start_x, start_y, width):
    def generate_bezier_curve(points, num_steps=50):
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
    num_images: int,
    training_folder: str,
    force_generate: bool = False,
    apply_brightness: bool = True,
    apply_noise: bool = True,
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
        print(f"Generating {num_images} type2 images starting at index {start_index}.")
    else:
        images_to_generate = num_images
        print(f"Generating {num_images} type2 images starting at index {start_index}.")

    for i in range(images_to_generate):
        image_number = start_index + i + 1
        image_filename = f"barcode_{image_number}.png"
        create_lbc_document(training_folder, image_filename, DOC_WIDTH, DOC_HEIGHT)
        print(f"Generated type2 image {i+1}/{images_to_generate}")


# Reuse other functions from doctype1.py:
# - add_scan_effects()
# - create_brightness_map()
# - generate_training_images() (with modified create_document function)
# - add_noise()
# - apply_damaged_pixels()

if __name__ == "__main__":
    generate_training_images(
        5, "training", force_generate=True, apply_brightness=True, apply_noise=True
    )
    print("LBC document generation complete.")
