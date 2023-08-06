# python train.py --img 416 --batch 16 --epochs 150 --data Basketball-Game-Object-Detection-1\data.yml --weights yolov5s.pt --cache

# img: define input image size
# batch: determine batch size
# epochs: define the number of training epochs (3000+ is common)
# data: dataset location is saved in the dataset.location
# weights: specify a path to weights to start transfer learning from. here we choose the generic COCO pretrained checkpoint
# cache images for faster training


#https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov5-object-detection-on-custom-data.ipynb#scrollTo=ZZ3DmmGQztJj

from roboflow import Roboflow


rf = Roboflow(api_key="4Hb4fBFqcjhIHDKIY8kk")
project = rf.workspace("matthew-hong-fnysh").project("basketball-game-object-detection")
dataset = project.version(1).download("yolov5")

