from multiprocessing import freeze_support
from ultralytics import YOLO
import cv2
import torch

model = YOLO("path/to/model")

video_path = "path/to/video" # *********** Replace with video *************

cap = cv2.VideoCapture(video_path)

results = model.track(source=video_path, conf=0.3, iou=0.5, show=True, save = True)

print("Number of animals: " + str((torch.max(results[-1].boxes.id).cpu())))

cap.release()