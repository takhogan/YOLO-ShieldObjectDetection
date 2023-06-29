from ultralytics import YOLO


if __name__ == '__main__':
    model_v8 = YOLO('yolov8m.pt')
    model_v8.train(data='data.yaml', epochs=100, imgsz=640, device='cpu')