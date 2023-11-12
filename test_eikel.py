#!/usr/bin/env python3

import cv2
from ultralytics import YOLO

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None and height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

model = YOLO('runs/detect/train4/weights/best.pt')

img_name = "IMG_9697"
top_path = "datasets/EikelboomSavanna/"
img_path = "{top_path}/images"
label_path = "{top_path}/labels"
img = f"{img_path}/{img_name}.JPG"
label_file = f"{label_path}/{img_name}.txt"


res = model.predict(img)#, imgsz=(3452,5184))

print(res)

annotated_frame = res[0].plot()

out_img = ResizeWithAspectRatio(annotated_frame, width=1024)

print(out_img.shape)


cv2.imshow("YOLOv8 prediction w/ partial trained model", out_img)
cv2.waitKey(5000)
#cv2.destroyAllWindows()
