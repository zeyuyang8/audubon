import torch
import torchvision

def get_model(chioce='fasterrcnn_resnet50_fpn'):
    if chioce == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    return model

def train_model(model, optimizer, train_loader, test_loader, n_epochs, device):
    ''' Train a model '''
    model = model.to(device)
    for epoch in range(n_epochs):
        print("Epoch:", epoch)
        model.train()
        epoch_loss = 0
        for batch, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{key: val.to(device) for key, val in target.items()} for target in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            print("Batch loss:", losses)
            epoch_loss += losses

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print("Loss:", epoch_loss)