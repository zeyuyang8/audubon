''' 
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 
torchvision.models.detection.faster_rcnn 
'''

import torch
from PIL import Image
from utils.global_const import COL_NAMES
from utils.data_processing import csv_to_df
from utils.data_processing import coordinate_to_box


class BirdDataset(torch.utils.data.Dataset):
    ''' Container for bird dataset '''
    def __init__(self, files, transforms=None):
        self.img_files = files['jpg']
        self.csv_files = files['csv']
        self.transforms = transforms

    def __getitem__(self, idx):
        # file path
        img_path, csv_path = self.img_files[idx], self.csv_files[idx]
        
        # image
        img = Image.open(img_path).convert("RGB")
        
        # boxes
        box_frame = csv_to_df(csv_path, COL_NAMES)
        num_objs = len(box_frame)
        boxes = []
        for row_idx in range(num_objs):
            x_1 = box_frame.iloc[row_idx]['x']
            y_1 = box_frame.iloc[row_idx]['y']
            width = box_frame.iloc[row_idx]['width']
            height = box_frame.iloc[row_idx]['height']
            boxes.append(coordinate_to_box(x_1, y_1, width, height))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # bird only detector, there is only one class 
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # target
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.img_files)
    
def collate_fn(batch):
    ''' Important for object detection '''
    return tuple(zip(*batch))