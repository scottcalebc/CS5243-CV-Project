#!/usr/bin/env python3


from ultralytics import YOLO


model = YOLO("yolov8n.pt")

results = model.track(source="https://www.youtube.com/watch?v=c0FtiZUO9Kg", show=True)
