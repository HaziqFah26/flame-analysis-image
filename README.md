# Image analysis to classify burning behaviour by flame position, luminosity, and colour
## Introduction
A code that utilize Ultralytics' yolo v8 to process experimental result videos on the burning rates of liquid fuel. The code will save the height of the burner flame and the BGR color value. 

## The following packages are required to run the code
    ultralytics
    roboflow
    CUDA (if using GPU for acceleration)

Install the recommended packages before runnning the code. Use commands in the command prompt windows or the python editor to install the packages using !pip install xxx.

`!pip install ultralytics`

`!pip install roboflow`

## Steps to train on custom dataset
1. Install YOLOv8
2. Roboflow Universe
3. Preparing a custom dataset
4. Custom Training
5. Validate Custom Model
6. Application with Custom Model

## Custom data
Used roboflow to annotate the burner flame images.
Sample notebook show how we can  add the Roboflow workflow project using API to download the annotated dataset to train the model.
Use the  below code to download the datset:
    
    from roboflow import Roboflow
    rf = Roboflow(api_key="xxxxxxxxxxxxxxxx")
    project = rf.workspace("custom-thxhn").project("fire-wrpgm")
    dataset = project.version(8).download("yolov8")

The custom dataset is available here:

    https://universe.roboflow.com/flame-detect/fire-detect-svial

## Training Model

## Evaluation
The below chart show  the loss , mAP (mean Average Precision) score for the train, test, validation set.

![Alt Text](https://github.com/HaziqFah26/flame-analysis-image/blob/main/training%20models/flame%20normal%20dataset%20epo%20100/results.png)

### Confusion Matrix : 
![Alt Text](https://github.com/HaziqFah26/flame-analysis-image/blob/main/training%20models/flame%20normal%20dataset%20epo%20100/confusion_matrix.png)

## Run
Run either the `Fire_Detection.jpynb` in jupyter notebook or `flame detect.py` in any python editor program. Make sure the necessary packages are installed and the proper files are in the same folder.

The model used is `best.pt` which is the result of training the model on the custom dataset. This is available at:

    https://uoe-my.sharepoint.com/:u:/g/personal/s2016278_ed_ac_uk/ERX--PRjkfVGni1fgSEWquYBCQJs9kdJki7P-DJTn_Jq8Q?e=2GqtTf

Enter the videos name/path (`1150_5.mp4` or `Clips/Clips/1150_.mp4`) and wait for the code to run.

The detection video is saved in `runs/detect/predict` and the results is saved in their respective folder name. The result file contains the BGR pixel color of the flame, the plot of height, the normal distribution of the height, the height array, the BGR arrays, and the conclusion results.

## Result
Example video: 1150_2.mp4

The resulted video is saved in `runs/detect/predict` named 1150_2.avi

### The plot of height
![Alt Text](https://github.com/HaziqFah26/flame-analysis-image/blob/main/Results/1150_2%20Data%20Results/Plot%20of%20height(1150_2).png)

### The normal distribution
![Alt Text](https://github.com/HaziqFah26/flame-analysis-image/blob/main/Results/1150_2%20Data%20Results/Normal%20Distribution%20(1150_2).png)

The arrays are saved in `Results/1150_2 Data Results`
