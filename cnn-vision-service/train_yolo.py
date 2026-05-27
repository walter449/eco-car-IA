from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="datasets/smoke_detection/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="smoke_detector"
)