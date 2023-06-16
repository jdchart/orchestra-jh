from PIL import Image
import os
from pathlib import Path
import json

def export_data_to_file(dataIn, path):
    f = open(path, "w")
    f.write(str(dataIn))
    f.close()

def export_image_region(imgPath, outPath, region):
    img = Image.open(imgPath)
    box = (region[0], region[1], region[2], region[3])
    img2 = img.crop(box)
    img2.save(outPath)

def makeDirsRecustive(pathList):
    for item in pathList:
        if os.path.isdir(item) == False:
            path = Path(item)
            path.mkdir(parents=True)

def writeJson(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def readJson(path):
    with open(path, 'r') as f:
        return json.load(f)