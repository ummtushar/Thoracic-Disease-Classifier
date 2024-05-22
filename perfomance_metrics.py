from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report, RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from train_test import test_model
import torch
from net import Net
from batch_sampler import BatchSampler
from image_dataset import ImageDataset
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

# NOTE: File used in the beginning of the project. Please ignore!

def ConfusionMatrix(y_pred, y):
    # Obtaining the predicted data
    y_pred = y_pred.cpu()
    y = y.cpu()
    reshaped = y.reshape(-1)

    # Plot Confusion Matrix
    report = classification_report(y, y_pred, zero_division=1)
    print(report)
    conf = confusion_matrix(reshaped, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf)

    FP = conf.sum(axis=0) - np.diag(conf)
    FN = conf.sum(axis=1) - np.diag(conf)
    TP = np.diag(conf)
    TN = conf.sum() - (FP + FN + TP)

    return disp, FP, FN, TP, TN



def ROC(y_pred_prob, y_pred, y):
    prob_reshape = y_pred_prob.cpu().reshape(-1)
    y_pred = y_pred.cpu()
    reshaped = y.cpu().reshape(-1)
    y_pred_prob = y_pred_prob.cpu().numpy()  # Convert to NumPy array
    y_pred = y_pred.cpu().numpy()            # Convert to NumPy array
    y = y.cpu().numpy()

    

    binary = []
    for i in range(len(y_pred)):
        if (y_pred[i] == reshaped[i]):
            binary.append(1)
        else:
            binary.append(0)
    fpr, tpr, threshold = metrics.roc_curve(binary, prob_reshape[:86])
    roc_auc = metrics.auc(fpr, tpr)
    disp_roc = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    return disp_roc


# def ROC2(y_train, y_test, y_score):
#     unique, counts = np.unique(np.concatenate((y_train, y_test)), return_counts=True)
#     print(dict(zip(unique, counts)))

#     label_binarizer = LabelBinarizer().fit(y_train)

    
#     y_onehot_test = label_binarizer.transform(y_test)
#     n_classes = len(label_binarizer.classes_)

#     class_off_interest = 1
#     class_id = np.flatnonzero(label_binarizer.classes_ == class_off_interest)[0]

#     fig, ax = plt.subplots(figsize=(6, 6))
#     target_names = ["Atelectasis", "Effusion", "Infiltration", "No Finding", "Nodule", "Pneumothorax"]
#     colors = cycle(["purple", "darkorange", "cornflowerblue", "red", "green", "darkblue"])
#     for class_id, color in zip(range(n_classes), colors):
#         RocCurveDisplay.from_predictions(
#             y_onehot_test[:, class_id],
#             y_score[:, class_id],
#             name=f"ROC curve for {target_names[class_id]}",
#             color=color,
#             ax=ax
#         )

#     plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")

#     return fig

# def ROC2(y_true, y_pred_prob, n_classes):
#     lb = LabelBinarizer()
#     y_true_binarized = lb.fit_transform(y_true)  # Binarize y_true
#     print("Shape of y_pred_prob:", y_pred_prob.shape)


#     fig, ax = plt.subplots(figsize=(8, 6))  # Prepare a figure for plotting

#     # Iterate over each class to calculate ROC
#     for i in range(n_classes):
#         y_true_class = y_true_binarized[:, i]  # True labels for class i
#         y_pred_class = y_pred_prob[:, i]       # Predicted probabilities for class i

#         # Calculate ROC curve
#         fpr, tpr, thresholds = roc_curve(y_true_class, y_pred_class)
#         roc_auc = auc(fpr, tpr)

#         # Plot ROC curve
#         RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)

#     plt.title("Multiclass ROC Curve")
#     plt.show()

#     return fig

def ROC_multiclass(y_true, y_pred_prob, n_classes):
    # Binarize the output
    y_true = label_binarize(y_true, classes=[*range(n_classes)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure()
    colors = cycle(['blue', 'red', 'green', 'yellow', 'orange', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC')
    plt.legend(loc="lower right")
    plt.show()

    return plt
