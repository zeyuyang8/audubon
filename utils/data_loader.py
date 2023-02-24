''' 
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 
torchvision.models.detection.faster_rcnn 
'''

import torch
import torch.utils.data
from PIL import Image
from global_const import COL_NAMES
from utils.data_processing import csv_to_df
from utils.data_processing import coordinate_to_box

class BirdDataset(torch.utils.data.Dataset):
    ''' Container for bird dataset '''
    def __init__(self, files, transforms=None):
        self.img_files = files['jpg']
        self.csv_files = files['csv']
        self.transforms = transforms

    def __getitem__(self, idx):
        # image
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")

        # annotations
        csv_path = self.csv_files[idx]
        frame = csv_to_df(csv_path, COL_NAMES)
        num_objs = len(frame)
        boxes = []
        for row_idx in num_objs:
            x_1 = frame.iloc[row_idx]['x']
            y_1 = frame.iloc[row_idx]['y']
            width = frame.iloc[row_idx]['width']
            height = frame.iloc[row_idx]['height']
            boxes.append(coordinate_to_box(x_1, y_1, width, height))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {}
        target['boxes'] = boxes
        # target['labels'] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        pass