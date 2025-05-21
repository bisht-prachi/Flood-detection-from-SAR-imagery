import torch
import tqdm
import numpy as np

def get_pred(device, model, test_loader, final_predictions):
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # load image and mask into device memory
            image = batch['image'].to(device)
            pred = model(image)
            output_tensor = pred#['out']

            # compute class predictions, i.e. flood or no-flood
            class_pred = output_tensor.argmax(dim=1)

            # convert class prediction to numpy
            class_pred = class_pred.detach().cpu().numpy()

            # add to final predictions
            final_predictions.append(class_pred.astype('uint8'))

    final_predictions = np.concatenate(final_predictions, axis=0)

    print('Predictions generated, shape:')
    # check final prediction shape
    print(final_predictions.shape)
    

    return final_predictions