from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

model(source="./videos/car-out/IMG_0643.MOV", conf=0.3, iou=0.5, show=True)