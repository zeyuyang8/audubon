# Audubon

##Project description


## Installation instructions
The Python version is 3.8.16 and I installed required packages using the following commands.
```
pip install numpy
pip install pandas
pip install matplotlib
pip install tqdm
pip3 install torch torchvision 
```

## Dataset
The dataset should contain a list of JPG image files and CSV files.

## Usage instructions
Open the [Colab link](https://colab.research.google.com/drive/1ogZnN_sZZRnXpQUwGrUWN_TZSoqCccks?usp=sharing) and run the notebook.

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
