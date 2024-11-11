from ultralytics import YOLO
from ultralytics.models import RTDETR

if __name__ == '__main__':
    model = YOLO(r'D:\deeplearning\PycharmProjects\copy_yolov8\ultralytics\cfg\models\v8\snu77\yolov8n-fasternet-mlca-siou-carafe.yaml')  # build a new model from YAML
    model.train(
        data='ultralytics/cfg/datasets/sheep_dataset2.yaml',
        epochs=1,
        imgsz=640,
        batch=1,
        save_period=5,
        resume=True,
        patience=200,
        project='runs/5-fold_compare/train/',
    )

