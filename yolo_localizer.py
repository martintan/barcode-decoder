import cv2
import torch
from ultralytics import YOLO

from constants import DOC_HEIGHT, DOC_WIDTH


def setup_yolo_detector(doc_width: int, doc_height: int) -> YOLO:
    model = YOLO("yolov8n.pt")
    model.model.yaml["nc"] = 1
    model.model.yaml["input_size"] = (doc_width, doc_height)
    return model


def train_yolo_detector(model: YOLO, doc_width: int, doc_height: int) -> None:
    model.train(
        data="barcode.yaml",
        epochs=50,
        imgsz=(doc_width, doc_height),
        batch=64,
        workers=2,
        lr0=0.001,
        patience=0,
        device="mps",  # apple m1/m2/m3
    )


def draw_top_boxes(image, boxes, num_boxes=5):
    sorted_boxes = sorted(boxes, key=lambda x: x.conf, reverse=True)[:num_boxes]
    for i, box in enumerate(sorted_boxes, 1):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(
            f"Barcode {i} localized at: x={x}, y={y}, width={w}, height={h}. conf: {box.conf.item()}"
        )
    return image


def localize_barcode_in_image(model: YOLO, image_path: str) -> None:
    sample_image = cv2.imread(image_path)

    if sample_image is None:
        print(f"Error: Unable to read image from {image_path}")
        return

    # Resize the image to match the model's input size
    sample_image_resized = cv2.resize(sample_image, (DOC_WIDTH, DOC_HEIGHT))

    # Localize the barcode
    results = model(sample_image_resized, conf=0.01)

    if len(results[0].boxes) > 0:
        sample_image_resized = draw_top_boxes(
            sample_image_resized, results[0].boxes, num_boxes=1
        )

        # Save the result
        cv2.imwrite("samples/localized_barcode.jpg", sample_image_resized)
        print("Localized image saved as 'localized_barcode.jpg'")
    else:
        print("No barcode detected in the image.")


def load_yolo_model(file_path: str) -> YOLO:
    model = YOLO(file_path)
    print(f"Model loaded from {file_path}")
    return model
