from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="belt.yaml",
    epochs=20,
    imgsz=640
)