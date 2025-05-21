import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp

#### DL Model

## model
def get_model(num_classes=2):

  model = smp.Unet(
      encoder_name="timm-efficientnet-b0",  # Specify the EfficientNet backbone
      encoder_weights="imagenet",     # Use pre-trained weights
      in_channels=3,                  # Input image channels (RGB)
      classes=2,                      # Number of output classes (flood/no-flood)
  )
  return model


# function to calculate iou score
def iou_score(preds, targets, num_classes=2):
    preds = torch.argmax(preds, dim=1)
    preds = preds.flatten().cpu().numpy()
    targets = targets.flatten().cpu().numpy()

    # If there's no positive class in ground truth, skip this sample
    if np.sum(targets == 1) == 0:
        return None  # skip IoU computation

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        iou = tp / (tp + fp + fn + 1e-6)
    else:
        iou = 0

    return iou


# training loop
def train_model(device, model, train_loader, dev_loader, num_epochs, iou_score, optimizer, scheduler, criterion):
    for epoch in range(num_epochs):
        if device == 'cuda':
            torch.cuda.empty_cache()

        # ===== TRAINING =====
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        count = 0

        pbar_train = tqdm(train_loader, desc=f"[Train Epoch {epoch+1}]")
        for batch in pbar_train:

        # load images and mask to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            #set up the NN
            optimizer.zero_grad()
            outputs = model(images)#['out'] #forward pass
            loss = criterion(outputs, masks)
            loss.backward() #backward pass
            optimizer.step()

            train_loss += loss.item()
            batch_iou = iou_score(outputs, masks)

            if batch_iou is not None:
                train_iou += batch_iou
                count += 1

            # update tqdm bar
            pbar_train.set_postfix(loss=loss.item(), iou=train_iou / (pbar_train.n + 1))

        train_loss /= max(count, 1)
        train_iou /= len(train_loader)

        #step scheduler after training epoch
        scheduler.step()

        # ===== VALIDATION =====
        model.eval()
        dev_loss = 0.0
        dev_iou = 0.0
        count = 0

        pbar_val = tqdm(dev_loader, desc=f"[Val Epoch {epoch+1}]")
        with torch.no_grad():
            for batch in pbar_val:

                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = model(images)#['out']

                loss = criterion(outputs, masks)

                batch_iou = iou_score(outputs, masks)

                if batch_iou is not None:
                    dev_iou += batch_iou
                    count += 1

                dev_loss += loss.item()

                # update tqdm bar
                pbar_val.set_postfix(loss=loss.item(), iou=dev_iou / (pbar_val.n + 1))

        dev_loss /= len(dev_loader)
        dev_iou /= max(count, 1)  # avoid division by zero


        print(f"\nEpoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | "
            f"Val Loss: {dev_loss:.4f}, Val IoU: {dev_iou:.4f}\n")
        
        
    return model

