from multiprocessing import freeze_support
from ultralytics import YOLO
import cv2
import supervision as sv

# Load a model, boxes, and dictionary
model = YOLO("D:/CS CV/runs/detect/snow_leopard2/weights/best.pt")  # new YOLOv8n model


box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=1, text_thickness=1, text_scale=.5)
CLASS_NAMES_DICT = model.model.names

print(CLASS_NAMES_DICT)

# image file
pic = 'D:/CS CV/test_data/snow_leopard/kim-murray-snow-leopard-mtn-murmer.jpg' # *********** Replace with image *************
img = cv2.imread(pic) 
results = model.predict(img, conf=.5)
boxes = []
for r in results:
    boxes = r.boxes.cpu().numpy()
# detection info
detections = sv.Detections(xyxy=results[0].boxes.xyxy.cpu().numpy(), confidence=results[0].boxes.conf.cpu().numpy(), class_id=results[0].boxes.cls.cpu().numpy().astype(int)) 
labels = []
for d in detections:
    labels.append(f"{d[2]:0.2f}")

print(len(detections))

img = box_annotator.annotate(scene = img, detections = detections, labels = labels)
cv2.imshow("pic", img)
cv2.waitKey(0) 