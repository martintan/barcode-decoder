import PIL
from PIL import Image, ImageDraw
import random
from datetime import datetime, timedelta
from utils import (
    add_text_to_image,
    apply_blur_effect,
    apply_pixel_damage_effect,
    draw_horizontal_line,
    draw_vertical_line,
    generate_random_text,
    generate_signature_scribble,
    generate_training_images,
    load_all_fonts,
    np_to_pil_grayscale,
    pil_to_np_grayscale,
    save_pil_jpeg,
    add_barcode,
    create_yolo_label,
)


def add_text_and_lines(
    draw: PIL.ImageDraw.Draw,
    doc_width: int,
) -> None:
    font_small, font_medium, font_large, font_extra_large = load_all_fonts()

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
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
        (generate_random_text(2), generate_random_text(1, max_word_length=3)),
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

    main_y = 120
    field_spacing = 20

    # Dispatch date and pickup date
    date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime(
        "%m/%d/%y"
    )
    add_text_to_image(draw, f"Dispatch Date:", (20, main_y), font_small)
    draw_horizontal_line(main_y + 15, 100, 300)
    add_text_to_image(draw, f"PICK UP DATE {date}", (120, main_y), font_medium)

    # Merchant details with lines
    merchant_y = main_y + 30
    add_text_to_image(draw, f"Merchant Ref. No:", (20, merchant_y), font_small)
    draw_horizontal_line(merchant_y + 15, 120, 400)
    add_text_to_image(draw, generate_random_text(8), (120, merchant_y), font_medium)

    add_text_to_image(
        draw, f"Merchant Name:", (20, merchant_y + field_spacing), font_small
    )
    draw_horizontal_line(merchant_y + field_spacing + 15, 120, 400)
    add_text_to_image(
        draw, generate_random_text(12), (120, merchant_y + field_spacing), font_medium
    )

    # Consignee details with lines
    consignee_y = merchant_y + field_spacing * 2
    add_text_to_image(draw, f"Consignee:", (20, consignee_y), font_small)
    draw_horizontal_line(consignee_y + 15, 100, 400)
    add_text_to_image(draw, generate_random_text(10), (100, consignee_y), font_medium)

    add_text_to_image(draw, f"Address:", (20, consignee_y + field_spacing), font_small)
    draw_horizontal_line(consignee_y + field_spacing + 15, 100, 400)
    add_text_to_image(
        draw, generate_random_text(10), (100, consignee_y + field_spacing), font_medium
    )

    # Add delivery attempt table
    table_y = main_y + 110
    table_headers = [
        generate_random_text(2),
        generate_random_text(2),
        generate_random_text(2),
        generate_random_text(2),
        generate_random_text(2),
        generate_random_text(2),
        generate_random_text(2),
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
            draw_vertical_line(draw, current_x, table_y, table_y + (row_height * 4))

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
    add_text_to_image(draw, "Signature over printed name", (20, sig_y), font_small)
    add_text_to_image(draw, "(First Name / Last Name)", (20, sig_y + 10), font_small)

    # Second signature
    draw_horizontal_line(sig_y, 250, 350)
    generate_signature_scribble(draw, 250, sig_y, 100)
    add_text_to_image(draw, generate_random_text(3), (250, sig_y), font_small)

    # Third signature
    draw_horizontal_line(sig_y, 400, 580)
    generate_signature_scribble(draw, 400, sig_y, 180)
    date = datetime.now().strftime("%m/%d/%y")
    add_text_to_image(draw, "Date", (400, sig_y), font_small)


def generate_image(
    training_folder: str,
    filename: str,
    doc_width: int,
    doc_height: int,
) -> None:
    background = Image.new("RGB", (doc_width, doc_height), color="white")
    draw = ImageDraw.Draw(background)
    barcode_x = (doc_width - 200) // 2
    background, barcode_dims = add_barcode(
        background,
        doc_width,
        doc_height,
        position=(barcode_x, 20),
    )
    create_yolo_label(training_folder, filename, doc_width, doc_height, barcode_dims)
    add_text_and_lines(draw, doc_width, doc_height)
    barcode_path = f"{training_folder}/images/{filename.rsplit('.', 1)[0]}.jpg"
    background.save(barcode_path, format="JPEG", quality=95)
    add_scan_effects(barcode_path, apply_brightness=False, apply_noise=False)


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
    print("Document Type 2 generation complete.")
