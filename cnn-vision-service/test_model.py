from ultralytics import YOLO

model = YOLO("trained_models/smoke_detector_v2.pt")

results = model(
    "test.jpg",
    conf=0.10,
    save=True
)

for result in results:
    print("Detecciones:", len(result.boxes))

    for box in result.boxes:
        print(
            "Confianza:",
            float(box.conf[0])
        )

        from ultralytics import YOLO

model = YOLO("trained_models/smoke_detector_v2.pt")

results = model("test.jpg")

for result in results:
    result.show()