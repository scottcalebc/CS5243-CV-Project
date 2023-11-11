#!/usr/bin/env python3

import os
import cv2
from ultralytics import YOLO


model = YOLO('runs/detect/train6/weights/last.pt')

res = model.predict("datasets/EikelboomSavanna/train/images/IMG_9708.JPG", imgsz=(3452,5184))

annotated_frame = res[0].plot()

cv2.imshow("YOLOv8 prediction w/ partial trained model", annotated_frame)
