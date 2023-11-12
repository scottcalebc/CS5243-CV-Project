#!/usr/bin/env python3


from ultralytics import YOLO

model = YOLO('yolov8n.pt')


# Would like to train on default resolution of training images, but this requries lots of memory
#model.train(data="data/EikelboomSavanna.yaml", epochs=10, imgsz=(3452,5184))

# This trains on default size of 640, model.train performs resizing of input images automagically
model.train(data="data/EikelboomSavanna.yaml", epochs=10)



