from ultralytics import YOLO

model = YOLO('./runs/detect/train3/weights/last.pt')
model.train(data='./detector_dataset/data.yaml', epochs=100, imgsz=640, batch=24, save_period=3, workers=3, resume=True)
