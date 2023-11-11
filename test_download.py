#!/usr/bin/env python3


import os
from pathlib import Path
import shutil

from ultralytics.utils.downloads import safe_download, unzip_file


top_dataset_dir = "./datasets"
dataset_dir = os.path.join(top_dataset_dir, "EikelboomSavana")
Path()
urls = [
        #'https://data.4tu.nl/file/9b1a2fcb-930e-4cc5-b9f0-a381dd1c7206/f978a7a0-f2aa-4c2b-a663-1165b247b56a',
        'https://data.4tu.nl/ndownloader/items/9b1a2fcb-930e-4cc5-b9f0-a381dd1c7206/versions/1'
        ]

safe_download(urls[0], file="EikelboomSavana.zip", dir=top_dataset_dir, unzip=True, delete=True, curl=True, progress=False)

if os.path.exists(os.path.join(dataset_dir, "data.zip")):
    data_unzip_dir = unzip_file(os.path.join(dataset_dir, "data.zip"), dataset_dir)

    if data_unzip_dir != dataset_dir:
        shutil.copytree(data_unzip_dir, dataset_dir)
        shutil.rmtree(data_unzip_dir)
