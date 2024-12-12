from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pre-trained model
model.train(data="data.yaml", epochs=10, imgsz=640, batch=16)
