from ultralytics import YOLO
def detect_yolo(input=None):
    # model = YOLO("best.pt")
    model = YOLO("yolo11n.pt")
    results = model.predict(input, save=True)
    boxes = results[0].boxes.xyxy.cpu()
    return f"{boxes}"