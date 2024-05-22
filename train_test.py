from tqdm import tqdm
import torch
from net import Net
from batch_sampler import  BatchSampler
from torch.nn import functional as F
import numpy as np
from net import Net
from batch_sampler import  BatchSampler
from torch.nn import functional as F
import numpy as np
import torch.nn as nn

from net import Net, ResNetModel, EfficientNetModel, EfficientNetModel_b7
from batch_sampler import BatchSampler
from image_dataset import ImageDataset

from typing import Callable, List, Tuple

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt



def train_model(
        # model: Net,  ## CHANGE NN HERE !
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    # Lets keep track of all the losses:
    losses = []
    # Put the model in train mode:
    model.train()
    # Feed all the batches one by one:
    for batch in tqdm(train_sampler):
        # Get a batch:
        x, y = batch
        # Making sure our samples are stored on the same device as our model:
        x, y = x.to(device), y.to(device)
        # Get predictions:
        predictions = model.forward(x)
        loss = loss_function(predictions, y)
        losses.append(loss)
        # We first need to make sure we reset our optimizer at the start.
        # We want to learn from each batch seperately,
        # not from the entire dataset at once.
        optimizer.zero_grad()
        # We now backpropagate our loss through our model:
        loss.backward()
        # We then make the optimizer take a step in the right direction.
        optimizer.step()
    return losses


def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
        fpr,
        tpr, 
        roc
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    # Setting the model to evaluation mode:
    model.eval()
    losses = []
    all_y_pred_probs = []
    all_y_true = []

    # We need to make sure we do not update our model based on the test data:
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            # Making sure our samples are stored on the same device as our model:
            x = x.to(device)
            y = y.to(device)
            prediction = model.forward(x)
            loss = loss_function(prediction, y)
            losses.append(loss)
            probabilities = F.softmax(prediction, dim=1)
            all_y_pred_probs.append(probabilities.cpu().numpy())
            all_y_true.extend(y.cpu().numpy())

    y_pred_probs = np.concatenate(all_y_pred_probs, axis=0)
    y_true = np.array(all_y_true)

    # NOTE: Comment this for loop and uncomment the # ROC for binary class for loop in order to see the correct Binary ROC curve.
    # Compute ROC curve and ROC area for each class
    for i in range(6):  # 6 classes
        a, b, _ = roc_curve(y_true == i, y_pred_probs[:, i])
        fpr[i].extend(a)
        tpr[i].extend(b)
        roc[i] = auc(fpr[i], tpr[i])

    # # ROC for binary class
    # for i in range(2):  # For binary classification
    #     fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_probs[:, i])
    #     roc[i] = auc(fpr[i], tpr[i])

    return losses, y_pred_probs