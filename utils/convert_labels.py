#!/usr/bin/env python3


import os
import subprocess
import pybboxes as pbx
import cv2


base_dir = "./data/Eikelboom"
annotations_file = "annotations_images.csv"

label_ids = []
with open(annotations_file, "r") as f:
    for annotation in f:
        # csv format:
        #   IMG_FILE, X1, Y1, X2, Y2, Category
        img_file, x1, y1, x2, y2, cat = annotation.split(",")

        if img_file == "FILE":
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        if cat not in label_ids:
            label_ids.append(cat)

        # YOLO annotation expects a index for the labels, build label index on the fly by first appearance of category
        label_id = label_ids.index(cat)


        # find file that matches annotation file
        paths = [line[2:] for line in subprocess.check_output(f"find . -iname '{img_file}'", shell=True).splitlines()]

        if len(paths) == 0:
            print("Error: Could not find image")
            continue

        orig_img = paths[0].decode()

        H, W, _ = cv2.imread(orig_img).shape

        img_type = orig_img.split("/")[0]

        new_annotation_file_name = img_file.split(".")[0] + ".txt"

        xc, yc, w, h = pbx.convert_bbox([x1, y1, x2, y2], from_type="voc", to_type="yolo", image_size=(W,H))

        #print(f"Generating new annotaions file: {img_type}/labels/{new_annotation_file_name}")


        with open(f"{img_type}/labels/{new_annotation_file_name}", "a") as af:
            af.write(f"{label_id} {xc} {yc} {w} {h}\n")



print("Generated labels:")
for i in range(len(label_ids)) :
    print(f"{label_ids[i]} : {i}")


        
        
