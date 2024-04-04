# Image analysis to classify burning behaviour by flame position, luminosity, and colour
## Introduction
A code that utilize Ultralytics' yolo v8 to process experimental result videos on the burning rates of liquid fuel. The code will save the height of the burner flame and the BGR color value. 

## The following packages are required to run the code
    ultralytics
    roboflow
    CUDA (if using GPU for acceleration)

Install the recommended packages before runnning the code. Use commands in the command prompt windows or the python editor to install the packages using !pip install xxx.

## Steps to train on custom dataset
1. Install YOLOv8
2. CLI Basics
3. Inference with Pre-trained COCO Model
4. Roboflow Universe
5. Preparing a custom dataset
6. Custom Training
7. Validate Custom Model
8. Inference with Custom Model

## Custom data
Used roboflow to  annotate fire and smoke images.
Sample notebook show how we can  add the Roboflow workflow project using API to download the annotated dataset to train the model.
Use the  below code to download the datset:
    
    from roboflow import Roboflow
    rf = Roboflow(api_key="xxxxxxxxxxxxxxxx")
    project = rf.workspace("custom-thxhn").project("fire-wrpgm")
    dataset = project.version(8).download("yolov8")


 
https://uoe-my.sharepoint.com/:u:/g/personal/s2016278_ed_ac_uk/ERX--PRjkfVGni1fgSEWquYBCQJs9kdJki7P-DJTn_Jq8Q?e=2GqtTf
