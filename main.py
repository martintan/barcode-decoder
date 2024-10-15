import os
from typing import Tuple
from doctype1 import create_barcode_image
from utils import generate_random_number
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, ToTensor, Compose
from ultralytics import YOLO


def generate_training_images(num_images: int, output_folder: str) -> None:
    if os.path.exists(output_folder):
        for root, dirs, files in os.walk(output_folder, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

    os.makedirs(output_folder, exist_ok=True)

    for i in range(num_images):
        create_barcode_image(f"barcode_{i+1}.png")
        print(f"Generated image {i+1}/{num_images}")


def setup_yolo_detector(input_size: Tuple[int, int] = (640, 320)) -> YOLO:
    model = YOLO("yolov8n.pt")
    model.model.yaml["nc"] = 1
    model.model.yaml["input_size"] = input_size
    return model


# We trained our model with batch size 64 and subdivisions 2 with an initial learning rate of 0.001.
def train_yolo_detector(model: YOLO) -> None:
    model.train(
        data="barcode.yaml",
        epochs=50,
        imgsz=640,
        batch=64,
        workers=2,
        lr0=0.001,
        patience=0,
        device="mps",  # apple m1/m2/m3
    )


if __name__ == "__main__":
    num_training_images = 5
    output_folder = "training"
    generate_training_images(num_training_images, output_folder)
    print("Training image generation complete.")

    yolo_model = setup_yolo_detector()
    train_yolo_detector(yolo_model)
    print("YOLO detector training complete.")
