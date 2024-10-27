from PIL import Image, ImageDraw
import random
import PIL
import numpy as np
from barcode import Code128
from barcode.writer import ImageWriter
from constants import DOC_HEIGHT
from doctype1 import add_scan_effects
from utils import add_barcode, create_yolo_label, debug_draw_boxes, draw_horizontal_line, draw_vertical_line, generate_signature_scribble, generate_training_images, load_all_fonts, save_pil_jpeg, generate_random_text

def generate_tracking_number() -> str:
    """Generate tracking number in format: 12digits-N-4digits-4digits"""
    part1 = ''.join(str(random.randint(0, 9)) for _ in range(12))
    part2 = 'N'
    part3 = ''.join(str(random.randint(0, 9)) for _ in range(4))
    part4 = ''.join(str(random.randint(0, 9)) for _ in range(4))
    return f"{part1}-{part2}-{part3}-{part4}"

def add_text_and_lines(draw: PIL.ImageDraw.Draw, doc_width: int) -> None:
    font_small, font_medium, font_large, font_extra_large = load_all_fonts()

    # Add border rectangle
    draw.rectangle([(0, 0), (doc_width-1, DOC_HEIGHT-1)], outline="black")

    # Add random text rows below left barcode
    left_margin = 20
    start_y = 100  # Position below the barcode
    for i in range(6):  # Add 3 rows of random text
        random_line = generate_random_text(word_count=random.randint(3, 5), max_word_length=12)
        draw.text((left_margin, start_y + (i * 10)), random_line, font=font_small, fill="black")

    # Add "Delivery" header
    header_x = doc_width // 2
    draw.text((header_x, 20), generate_random_text(1, 8), font=font_large, fill="black", anchor="mm")

    # Add certification text
    draw.text((header_x, 60), generate_random_text(word_count=6), font=font_medium, fill="black", anchor="mm")
    draw.text((header_x, 75), generate_random_text(word_count=8), font=font_medium, fill="black", anchor="mm")
    draw.text((header_x, 90), generate_random_text(word_count=7), font=font_medium, fill="black", anchor="mm")

    # Add checkbox section - moved up and more compact
    checkbox_y = 115  # Moved up from 120
    draw.text((header_x - 100, checkbox_y), generate_random_text(2), font=font_medium, fill="black")
    # Draw checkboxes
    draw.rectangle([(header_x, checkbox_y), (header_x + 15, checkbox_y + 15)], outline="black")
    draw.text((header_x + 25, checkbox_y), generate_random_text(1), font=font_medium, fill="black")
    
    draw.rectangle([(header_x + 150, checkbox_y), (header_x + 165, checkbox_y + 15)], outline="black")
    draw.text((header_x + 175, checkbox_y), generate_random_text(2), font=font_medium, fill="black")

    # Add signature section text - reduced spacing
    cert_y = 140  # Moved up from 160
    line_spacing = 12  # Reduced from 15
    draw.text((header_x - 100, cert_y), generate_random_text(word_count=5), font=font_medium, fill="red")
    draw.text((header_x - 100, cert_y + line_spacing), generate_random_text(word_count=6), font=font_medium, fill="red")
    draw.text((header_x - 100, cert_y + line_spacing * 2), generate_random_text(word_count=6), font=font_medium, fill="red")
    draw.text((header_x - 100, cert_y + line_spacing * 3), generate_random_text(word_count=4), font=font_medium, fill="red")

    # Add signature and date lines side by side
    sig_y = 195  # Moved up from 220
    
    # Signature line (left side)
    draw_horizontal_line(draw, sig_y, header_x - 100, header_x + 50)
    draw.text((header_x - 80, sig_y + 5), generate_random_text(word_count=3), font=font_small, fill="black")
    draw.text((header_x - 80, sig_y + 15), generate_random_text(word_count=2), font=font_small, fill="black")
    draw.text((header_x - 80, sig_y + 25), generate_random_text(word_count=3), font=font_small, fill="black")

    # Date line (right side)
    draw.text((header_x + 100, sig_y - 15), generate_random_text(1), font=font_medium, fill="black")
    draw_horizontal_line(draw, sig_y, header_x + 100, header_x + 250)

    # Add large table at bottom
    table_y = DOC_HEIGHT - (DOC_HEIGHT // 3)  # Start at 2/3 down the page
    headers = [generate_random_text(2) for _ in range(5)]
    col_widths = [150, 150, 100, 100, 100]
    row_height = 15  # Reduced from 40 to 25
    num_data_rows = 4  # Explicitly define number of rows below header
    
    # Calculate table dimensions
    table_width = sum(col_widths)
    table_start_x = (doc_width - table_width) // 2
    
    # Draw outer table border
    draw.rectangle(
        [
            (table_start_x, table_y),
            (table_start_x + table_width, table_y + (row_height * (num_data_rows + 1))),  # +1 for header row
        ],
        outline="black",
    )

    # Draw column headers and vertical lines
    current_x = table_start_x
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        # Draw vertical lines
        draw_vertical_line(
            draw,
            current_x,
            table_y,
            table_y + (row_height * (num_data_rows + 1))
        )
        
        # Add header text
        draw.text((current_x + 10, table_y + 5), header, font=font_small, fill="black")
        current_x += width
    
    # Draw final vertical line
    draw_vertical_line(
        draw,
        current_x,
        table_y,
        table_y + (row_height * (num_data_rows + 1))
    )

    # Draw horizontal lines between rows (num_data_rows + 1 to include lines after header)
    for i in range(1, num_data_rows + 1):
        y = table_y + (row_height * i)
        draw_horizontal_line(
            draw,
            start_y=y,
            start_x=table_start_x,
            end_x=table_start_x + table_width,
        )

    # Add signatures only in first data row
    signature_y = table_y + row_height + 5  # Adjusted padding from 10 to 5 to fit in smaller row
    current_x = table_start_x
    for width in col_widths:
        generate_signature_scribble(
            draw,
            current_x + 10,
            signature_y,
            width - 20
        )
        current_x += width

def generate_image(
    training_folder: str,
    filename: str,
    doc_width: int,
    doc_height: int,
) -> None:
    image = Image.new("RGB", (doc_width, doc_height), color="white")
    draw = ImageDraw.Draw(image)
    
    # Add main content
    add_text_and_lines(draw, doc_width)
    
    # Add long tracking barcode (left)
    tracking_number = generate_tracking_number()
    barcode = Code128(tracking_number, writer=ImageWriter())
    barcode_img = barcode.render(writer_options={
        "module_width": 0.11,
        "module_height": 2.4,
        "font_size": 2,
        "text_distance": 1,
        "foreground": "black",
        "center_text": True,
        "quiet_zone": 0,
    })
    image.paste(barcode_img, (20, 20))
    
    # Add small barcode (right) - this is the one we'll label
    barcode_x = doc_width - 150
    image, barcode_coords = add_barcode(image, doc_width, doc_height, position=(barcode_x, 20))
    
    # Add scan effects
    image, barcode_coords = add_scan_effects(image, barcode_coords=barcode_coords)
    
    debug_draw_boxes(image, barcode_coords)
    
    # Save image and create label
    image_path = f"{training_folder}/images/{filename.rsplit('.', 1)[0]}.jpg"
    save_pil_jpeg(image, image_path)
    create_yolo_label(training_folder, filename, doc_width, doc_height, barcode_coords)

if __name__ == "__main__":
    generate_training_images(
        generate_func=generate_image,
        num_images=5,
        training_folder="training",
        force_generate=True,
        start_index=0,
    )
    print("Document Type 3 generation complete.")
