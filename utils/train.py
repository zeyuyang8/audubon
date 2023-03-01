import torch
from livelossplot import PlotLosses

# Defining as global the device to use (by default CPU).
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train_model(batch_size, n_epochs, learningRate, model, 
                cost_function, optimizer,scheduler, train_loader, val_loader):

    # Move the model and cost function to GPU (if needed).
    model = model.to(device)
    cost_function = cost_function.to(device)

    # Keep track of best accuracy so far.
    best_accuracy = 0 
    liveloss = PlotLosses()

    # Main for loop of SGD.
    for epoch in range(0, n_epochs):
        logs = {}

        # initialize control variables.
        correct = 0
        cumulative_loss = 0
        n_samples = 0

        # Set the model in training mode.
        model.train()

        # Sample a batch on each iteration.
        for (batch_id, (xb, yb)) in enumerate(train_loader):
            model.zero_grad()

            # Move (x,y) data to GPU (if so desired).
            xb = xb.to(device)
            yb = yb.to(device)

            # Compute predictions.
            predicted = model(xb)
            # print(xb.shape, yb.shape, predicted.shape)

            # Compute loss.
            loss = cost_function(predicted, yb)
            cumulative_loss += loss.item()

            # Count how many correct in batch.
            predicted_ = predicted.detach().softmax(dim = 1)
            max_vals, max_ids = predicted_.max(dim = 1)
            correct += (max_ids == yb).sum().cpu().item()
            n_samples += xb.size(0)

            # Compute gradients (autograd).
            loss.backward()

            # Run one basic training step of SGD.
            optimizer.step()

            # Keep track of loss and accuracy for the plot.
            n_batches = 1 + batch_id 
            logs['loss'] = cumulative_loss / n_batches
            logs['accuracy'] = correct / n_samples
    
        # initialize control variables.
        correct = 0
        cumulative_loss = 0
        n_samples = 0

        # Set the model in evaluation mode.
        model.eval()

        # No need to keep track of gradients for this part.
        with torch.no_grad():
            # Run the model on the validation set to keep track of accuracy there.
            for (batch_id, (xb, yb)) in enumerate(val_loader):

                # Move data to GPU if needed.
                xb = xb.to(device)
                yb = yb.to(device)
            
                # Compute predictions.
                predicted = model(xb)

                # Compute loss.
                loss = cost_function(predicted, yb)
                cumulative_loss += loss.item()

                # Count how many correct in batch.
                predicted_ = predicted.detach().softmax(dim = 1)
                max_vals, max_ids = predicted_.max(dim = 1)
                correct += (max_ids == yb).sum().cpu().item()
                n_samples += xb.size(0)

                # Keep track of loss and accuracy for the plot.
                n_batches = 1 + batch_id
                logs['val_loss'] = cumulative_loss / n_batches
                logs['val_accuracy'] = correct / n_samples

        # Save the parameters for the best accuracy on the validation set so far.
        if logs['val_accuracy'] > best_accuracy:
            best_accuracy = logs['val_accuracy']
            torch.save(model.state_dict(), 'best_model_so_far.pth')

        # Update the plot with new logging information.
        liveloss.update(logs)
        liveloss.send()

        # What is this for? Please look it up.
        if scheduler != -1:
            scheduler.step()

# Load the model parameters for the one that achieved the best val accuracy.
# model.load_state_dict(torch.load('best_model_so_far.pth'))    