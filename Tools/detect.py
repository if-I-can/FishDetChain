from ultralytics import YOLO
def detect_yolo(input=None):
    model = YOLO("best.pt")
    results = model.predict(input, save=True)
    boxes = results[0].boxes.xywh.cpu()
    return f"{boxes}"