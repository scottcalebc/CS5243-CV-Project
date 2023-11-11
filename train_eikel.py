#!/usr/bin/env python3


from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data="data/EikelboomSavanna.yaml", epochs=10, imgsz=(3452,5184))



