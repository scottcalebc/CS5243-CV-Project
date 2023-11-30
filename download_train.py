from multiprocessing import freeze_support
from ultralytics import YOLO
import supervision as sv
from roboflow import Roboflow

# # Roboflow key
rf = Roboflow(api_key="I07srwVqExH7LvgRx8n5")

#  __       ___        __   ___ ___  __  
# |  \  /\   |   /\   /__  |__   |  /__  
# |__/ /--\  |  /--\   __/ |___  |   __/ 
                                     
# # COWS
# project = rf.workspace("university-of-texas-at-san-antonio").project("cows-mlu15")
# dataset = project.version(2).download("yolov8")

# # # SHARKS
# project = rf.workspace("utsa-1cehd").project("sharks-rrwfg")
# dataset = project.version(1).download("yolov8")

# # ZEBRAS
# project = rf.workspace("animal-detection-yvpsn").project("zebra-detect-3")
# dataset = project.version(1).download("yolov8")

# # ELEPHANT SEALS
# project = rf.workspace("w251elephantseal").project("elephant-seal-detection")
# dataset = project.version(2).download("yolov8")

# # SNOW LEOPARD
# project = rf.workspace("snowbars").project("bars-7sljf")
# dataset = project.version(5).download("yolov8")

# # # WHALE
# project = rf.workspace("university-of-texas-at-san-antonio").project("whales-hksa5")
# dataset = project.version(1).download("yolov8")

# # BOAR
# project = rf.workspace("jjyyppp-naver-com").project("wildboar-msble")
# dataset = project.version(3).download("yolov8")

# # WATER BUFFALO
# project = rf.workspace("aiamazon").project("deep_buffalo_rgb")
# dataset = project.version(1).download("yolov8")

# # SAVANAH ANIMALS
# project = rf.workspace("graduation-nnzal").project("animals-kapzz")
# dataset = project.version(3).download("yolov8")

# # rhino
# project = rf.workspace("spie24").project("xyz-rzfpc")
# dataset = project.version(2).download("yolov8")

# # SHEEP
# project = rf.workspace("riisprivate").project("sheepcounter")
# dataset = project.version(10).download("yolov8")

# ___  __              
#  |  |__)  /\  | |\ | 
#  |  |  \ /--\ | | \| 
                     
# Load a model, boxes, and dictionary
model = YOLO("yolov8n.yaml")  # new YOLOv8n model

# train model
if __name__ == '__main__':
    # # cows
    # result = model.train(data = "datasets\cows-2\data.yaml", epochs = 100, imgsz = 640, device = 0)
    
    # # sharks
    # result = model.train(data = "datasets\sharks\data.yaml", epochs=100, device = 0, batch = 16)

    # snow leopard
    result = model.train(data = "D:\CS CV\datasets\snow_leopard\data.yaml", epochs=300, device = 0, batch = 24)

    # # boar
    # result = model.train(data = "datasets/boar/data.yaml", epochs=100, device = 0, batch = 16)

    # # elephant seals
    # result = model.train(data = "datasets\elephant_seal\data.yaml", epochs=100, device = 0, batch = 16)

    # # savannah
    # result = model.train(data = "datasets\savannah\data.yaml", epochs=100, device = 0, batch = 16)

    # # water buffalo
    # result = model.train(data = "datasets\water_buffalo\data.yaml", epochs=100, device = 0, batch = 16)

    # # whale
    # result = model.train(data = "D:\CS CV\datasets\whales-1\data.yaml", epochs=300, device = 0, batch = 28)

    # # zebra
    # result = model.train(data = "datasets\zebra\data.yaml", epochs=100, device = 0, batch = 16)

    # # rhino
    # result = model.train(data = "D:/CS CV/datasets/rhino/data.yaml", epochs=100, device = 0, batch = 16)

    # # sheep
    # result = model.train(data = "D:/CS CV/datasets/sheep/data.yaml", epochs=100, device = 0, batch = 16)