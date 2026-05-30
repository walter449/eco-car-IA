from ultralytics import YOLO

model = YOLO("runs/detect/smoke_detector/weights/best.pt")

results = model(
    "test.jpg",
    conf=0.70
)

for result in results:
    result.save(filename="prediction.jpg")

    boxes = result.boxes

    for box in boxes:
        clase = int(box.cls[0])
        confianza = float(box.conf[0])

        print(
            f"Clase: {model.names[clase]} | "
            f"Confianza: {confianza:.2f}"
        )