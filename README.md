# Audubon

## Project description
Houston Audubon uses drone photography to monitor populations of colonial waterbirds in the Texas Gulf Coast. Researchers need to comb through drone images and manually count up birds by species class, which can take weeks for even one high resolution image. Seeking to automate this process, Houston Audubon partnered with the Data to Knowledge (D2K) lab at Rice University. Student teams have developed an object detection based deep learning model that can automatically detect birds within a UAV image and classify their species. This semester, we are continuing the project with two main objecitves:
  1. Improve species classification capabilities of the model.
  2. Develop an AI-assisted waterbird annotation tool.

## Dataset
Houston Audubon has collected 52 GB of raw images using a DJI M300RTK UAV with a P1 Camera attachment. The images are 8192 X 5460 pixels. For training and testing our model, Houston Audubon has provided us with a 4 GB subset of raw images with annotations for each bird.

Each annotated UAV image has a corresponding CSV file containing bird annotations. Each bird is encapsulated by a bounding box in the image, and each bounding box represents a row in the CSV file. Each row contains the following data about the bird within the bounding box:

  - Unique four letter bird class identifier 
  - Bird class name 
  - Smallest x coordinate in the bounding box (coordinates in terms of pixels)
  - Smallest y coordinate in the bounding box
  - Width of the bounding box
  - Height of the bounding box

## Folder structure
code 
├──

## Installation instructions
The Python version is 3.8.16 and users can install the required packages using the following commands.
```
pip install numpy
pip install pandas
pip install matplotlib
pip install tqdm
pip3 install torch torchvision 
```

## Usage instructions
Open the [Colab link](https://colab.research.google.com/drive/1wU5k5jI9TlPWy3CzXb4gabZ__YB-Cp97?usp=sharing) and run the notebook.

## TODO
- Understand the feature representations of Faster RCNN.
- Build a hierachical classifier.
- Build a ResNet multi-class classifier.
- Compare multi-class classifier with hierachical classifier.
- Learn https://lost.training/


## Comments
We can extract feature layers from the following types of model:

1. Bird-only detector
2. Visual-class detector (only big groups)
3. Multi-class detector (all species)
