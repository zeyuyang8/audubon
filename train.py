''' 
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 
torchvision.models.detection.faster_rcnn 
'''

import torch
from tqdm import tqdm
import torchvision
from PIL import Image
from torchvision.transforms import functional as F
from audubon.const import COL_NAMES
from audubon.utils.data_processing import csv_to_df
from audubon.utils.data_processing import coordinate_to_box


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
    
def bird_collate_fn(batch):
    ''' Important for object detection '''
    return tuple(zip(*batch))

def get_bird_dataloaders(train_files, test_files):
    # use our dataset and defined transformations
    trainset = BirdDataset(train_files, F.to_tensor)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=bird_collate_fn # important otherwise it raises an error
    ) 

    testset = BirdDataset(test_files, F.to_tensor)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=bird_collate_fn # important otherwise it raises an error
    ) 
    return trainloader, testloader

def get_model_and_optim(chioce='fasterrcnn_resnet50_fpn'):
    if chioce == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    return model, optimizer

def train_model(model, optimizer, trainloader, testloader, n_epochs, device):
    ''' Train a model '''
    model = model.to(device)
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for batch, (images, targets) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1} of {n_epochs}", leave=True, ncols=80)):
            images = list(image.to(device) for image in images)
            targets = [{key: val.to(device) for key, val in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print("Epoch:", epoch + 1, "| Loss:", epoch_loss)
        print()