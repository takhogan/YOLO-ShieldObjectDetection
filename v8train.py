from ultralytics import YOLO
import sys


if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'yolov8m.pt'

    print('starting training with model ', model_path)
    model_v8 = YOLO()
    model_v8.train(data='data.yaml', epochs=100, imgsz=640, device='cpu')