from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.to("cuda:0")

print(model.device)

model.train(data="config.yaml", epochs=3, imgsz=640, device=[0], workers=4, batch=4)
metrics = model.val()
results = model('/home/batiukmaks/Documents/Intelligent-Parking-Management-System/my-parking/dataset/valid/images/frame_111.jpg', imgsz=640, device=[0], workers=4, batch=4)
path = model.export(format='onnx')
