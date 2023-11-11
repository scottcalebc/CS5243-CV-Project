#!/usr/bin/env python3


import os
from pathlib import Path

from ultralytics.utils.downloads import download, unzip_file


dir = "./datasets/EikelboomSavana"
urls = [
        #'https://data.4tu.nl/file/9b1a2fcb-930e-4cc5-b9f0-a381dd1c7206/f978a7a0-f2aa-4c2b-a663-1165b247b56a',
        'https://data.4tu.nl/ndownloader/items/9b1a2fcb-930e-4cc5-b9f0-a381dd1c7206/versions/1'
        ]

download(urls, dir=dir, curl=True, threads=4)

# if os.path.exists(os.path.join(dir, "f978a7a0-f2aa-4c2b-a663-1165b247b56a")):
#     unzip_file(os.path.join(dir, "f978a7a0-f2aa-4c2b-a663-1165b247b56a"))
