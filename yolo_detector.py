from ultralytics import YOLO


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
