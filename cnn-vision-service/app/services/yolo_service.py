from ultralytics import YOLO

model = YOLO(
    "runs/detect/smoke_detector/weights/best.pt"
)

def predict(image_path):
    return model(image_path)