from ultralytics import YOLO
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()

    model = YOLO("yolov8n.pt")
    model.to("mps")

    print(model.device)

    model.train(data="config.yaml", epochs=3, imgsz=640, device="mps", workers=4, batch=4)
    metrics = model.val()
    results = model('dataset/valid/images/frame_111.jpg', imgsz=640, device="mps", workers=4, batch=4)
    path = model.export(format='onnx')
