# Custom imports
from batch_sampler import BatchSampler
from image_dataset import ImageDataset
from net import Net, ResNetModel, EfficientNetModel, EfficientNetModel_b7
from train_test import train_model, test_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
from visualise_performance_metrics import create_confusion_matrix, ROC_multiclass
from image_dataset_BINARY import ImageDatasetBINARY
from net_BINARY import Net_BINARY
# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary 

# Other imports
import matplotlib.pyplot as plt  
from matplotlib.pyplot import figure
import os
import argparse
import plotext  
from datetime import datetime
from pathlib import Path
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
import numpy as np

def main(args: argparse.Namespace, activeloop: bool = True) -> None:
    # NOTE: Uncomment the dataset you would like to use. For instance, if you would like to run the Binary Model, 
    # you would need to comment the ImageDataset Class lines, or viceversa if you are running the other models.
    # Load the train and test data set
    train_dataset = ImageDataset(Path('dc1/data/X_train.npy'), Path('dc1/data/Y_train.npy'))
    test_dataset = ImageDataset(Path('dc1/data/X_test.npy'), Path('dc1/data/Y_test.npy'))

    # Load the BINARY train and BINARY test data set 
    # train_dataset = ImageDatasetBINARY(Path('dc1/data/X_train.npy'), Path('dc1/data/Y_train.npy'))
    # test_dataset = ImageDatasetBINARY(Path('dc1/data/X_test.npy'), Path('dc1/data/Y_test.npy'))

    # Load the Neural Net. 
    # NOTE: set number of distinct labels here
    # NOTE: uncomment when you need to use one of the models
    # Improved Net
    # model = Net(n_classes=6) 
    # ResNet Pre-Trained Model
    # model = ResNetModel(n_classes=6)  
    # EfficientNet Model Pre-Trained == OUR SELECTED MODEL 
    model = EfficientNetModel(n_classes=6)
    # Binary Model 
    # model = Net_BINARY(n_classes=2)
    
    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # fpr = {x:[] for x in range(6)}
    # tpr = {x:[] for x in range(6)}
    # auc = {}

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif (
            torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)
    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Lets now train and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )

    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []

    for e in range(n_epochs):
        if activeloop:
            # Training:
            losses = train_model(model, train_sampler, optimizer, loss_function, device)
            # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            # Testing:
            # losses, y_pred_probs = test_model(model, test_sampler, loss_function, device)
            fpr = {x:[] for x in range(6)}
            tpr = {x:[] for x in range(6)}
            auc = {}
            
            # # Calculating and printing statistics:
            losses, y_pred_probs = test_model(model, test_sampler, loss_function, device, fpr, tpr, auc)

            # # Calculating and printing statistics:
            mean_loss = sum(losses) / len(losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            print(auc)

            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()


    ##################################################################################################################
    #                    R O C      C U R V E S
    # ##################################################################################################################        
    # NOTE: If you would like to run the ROC function for the Binary dataset, you would need to comment the following code 
    # and uncomment the # ROC CURVE FOR BINARY. Additionally, in train_test.py file, in order to make the Binary ROC curve 
    # you would need to comment some lines specified in the file. Please check the file.

    # ROC CURVE FOR MULTICLASS
    plt.figure(figsize=(8, 6))

    colors = plt.cm.get_cmap('viridis', 6).colors
    class_names = ['Class 0 (Atelactasis)','Class 1 (Effusion)', 'Class 2 (Infiltration)', 'Class 3 (No Finding)', 'Class 4 (Nodule)', 'Class 5 (Pneumonia)']

    for i, color in zip(range(6), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='{} (AUC = {:.2f})'.format(class_names[i], auc[i]))

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 6 Classes')
    plt.legend(loc="lower right")
    plt.show()

    # ROC CURVE FOR BINARY
    # for i in range(2):
    #     plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {auc[i]:.2f})')

    # plt.plot([0, 1], [0, 1], 'k--', label='Random chance')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves for 2 Classes (1=Sick, 0=Non-sick)')
    # plt.legend(loc='lower right')
    # plt.show()

            
    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))

    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}{now.day:02}{now.hour}_{now.minute:02}.txt")
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}{now.day:02}{now.hour}_{now.minute:02}.txt")

    # Create plot of losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()


    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}{now.day:02}{now.hour}_{now.minute:02}.png")

    ##################################################################################################################
    #      C O N F U S I O N      M A T R I X  &    C L A S S I F I C A T I O N      R E P O R T
    # ##################################################################################################################   
    true_labels = test_dataset.get_labels()

    # Set the model to evaluation mode
    model.eval()

    predicted_labels = []
    with torch.no_grad():
        for inputs, _ in test_dataset:
            inputs = inputs.unsqueeze(0).to(device)

            outputs = model(inputs)

            # Get predicted labels by getting max value (aka most likely)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())

    # Calculate Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print("Confusion Matrix:")
    print(conf_matrix)
    # plot the confusion matrix
    # fig, ax = plt.subplots()
    # ConfusionMatrixDisplay(confusion_matrix=conf_matrix).plot(ax=ax, cmap="Blues")
    # plt.show()
    # plt.savefig('confusion_matrix.png')
    create_confusion_matrix(true_labels, predicted_labels)

    # Classification report (accuracy, precision, f1 etc)
    class_report = classification_report(true_labels, predicted_labels)
    print("\nClassification Report:")
    print(class_report)

    fig.savefig(Path("artifacts") / f"session_{now.month:02}{now.day:02}{now.hour}_{now.minute:02}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=1, type=int)
    parser.add_argument("--batch_size", help="batch_size", default=25, type=int)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    main(args)