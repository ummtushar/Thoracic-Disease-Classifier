# # # # Imports
# # # import torch  
# # # import numpy as np
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # # Imports
# # # import torch  
# # # import numpy as np
# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns

# # # from sklearn.metrics import confusion_matrix, roc_curve, auc
# # # from typing import Callable, List, Tuple
# # # import torch.nn as nn
# # # from pathlib import Path
# # # import torch.nn.functional as F
# # # from yaml import FlowSequenceStartToken  
# # # from sklearn.metrics import confusion_matrix, roc_curve, auc
# # # from typing import Callable, List, Tuple
# # # import torch.nn as nn
# # # from pathlib import Path
# # # import torch.nn.functional as F
# # # from yaml import FlowSequenceStartToken  

# # Import files 
# from image_dataset import ImageDataset
# from net import Net, ResNetModel, EfficientNetModel
# from train_test import train_model, test_model
# from batch_sampler import BatchSampler

# NOTE: File used in the very beginning of the project. Please ignore!

# maincolor = '#4a8cffff'
# secondcolor = '#e06666'

# # Train data
# labels_train_path = 'dc1/data/Y_train.npy'
# data_train_path = 'dc1/data/X_train.npy'
# # Test data
# labels_test_path = 'dc1/data/Y_test.npy'
# data_test_path = 'dc1/data/X_test.npy'


# y_train = np.load(labels_train_path)
# unique_labels = np.unique(y_train)
# data_train = np.load(data_train_path)


# # Data Verification to check if we all have everything good
# data_shape = data_train.shape
# data_type = data_train.dtype
# labels_shape = y_train.shape
# labels_type = y_train.dtype
# print(f"Data Shape: {data_shape}, Data Type: {data_type}")
# print(f"Labels Shape: {labels_shape}, Labels Type: {labels_type}")

# # Check the range and distribution of features
# data_range = (np.min(data_train), np.max(data_train))

# # Label Encoding in accordance to the diseases
# class_names_mapping = {
#     0: 'Atelectasis',
#     1: 'Effusion',
#     2: 'Infiltration',
#     3: 'No Finding',
#     4: 'Nodule',
#     5: 'Pneumonia'
# }
 
# print("Unique classes in the training set:")
# for class_id in unique_labels:
#     print(f"Class ID {class_id}: {class_names_mapping[class_id]}")

# # df for distribution analysis
# df_data_range = pd.DataFrame(data_train.reshape(data_train.shape[0], -1))

# ###################################################################
# ###########   A D V A N C E D         A N L Y S I S     ###########
# ##################################################################

# # Y test data (labels)
# y_test = np.load(labels_test_path)

# # Initialize model (NET)
# n_classes = 6
# # NOTE : change the nn here! 
# model = Net(n_classes=n_classes)  
# # model = ResNetModel(n_classes=n_classes)
# # model = EfficientNetModel(n_classes=n_classes)

# # Device for test_model function call
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Initialize the loss function
# loss_function = nn.CrossEntropyLoss()  # we can use another, this one i found in internet but I was getting errors...


# # # Data Verification to check if we all have everything good
# # data_shape = data_train.shape
# # data_type = data_train.dtype
# # labels_shape = y_train.shape
# # labels_type = y_train.dtype
# # print(f"Data Shape: {data_shape}, Data Type: {data_type}")
# # print(f"Labels Shape: {labels_shape}, Labels Type: {labels_type}")

# # # Check the range and distribution of features
# # data_range = (np.min(data_train), np.max(data_train))

# # # Label Encoding in accordance to the diseases
# # class_names_mapping = {
# #     0: 'Atelectasis',
# #     1: 'Effusion',
# #     2: 'Infiltration',
# #     3: 'No Finding',
# #     4: 'Nodule',
# #     5: 'Pneumonia'
# # }
 
# # print("Unique classes in the training set:")
# # for class_id in unique_labels:
# #     print(f"Class ID {class_id}: {class_names_mapping[class_id]}")

# # # df for distribution analysis
# # df_data_range = pd.DataFrame(data_train.reshape(data_train.shape[0], -1))

# # ###################################################################
# # ###########   A D V A N C E D         A N L Y S I S     ###########
# # ##################################################################

# # # Y test data (labels)
# # y_test = np.load(labels_test_path)

# # # Initialize model (NET)
# # n_classes = 6
# # # NOTE : change the nn here! 
# # model = Net(n_classes=n_classes)  
# # # model = ResNetModel(n_classes=n_classes)
# # # model = EfficientNetModel(n_classes=n_classes)

# # # Device for test_model function call
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)

# # # Initialize the loss function
# # loss_function = nn.CrossEntropyLoss()  # we can use another, this one i found in internet but I was getting errors...

# # # Load test dataset w function
# # test_dataset = ImageDataset(Path("dc1/data/X_test.npy"), Path("dc1/data/Y_test.npy"))

# # # Initialize the BatchSampler 
# # batch_size = 32  
# # test_loader = BatchSampler(batch_size=batch_size, dataset=test_dataset, balanced=False)  #  'balanced' or not we can choose depending on what we want

# # # Function call
# # losses, predicted_labels, true_labels, probabilities = test_model(model, test_loader, loss_function, device)

# #####################  R O C     C U R V E   #####################
# def plot_multiclass_roc_curve(y_true, y_scores, num_classes):
#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
    
#     for i in range(num_classes):
#         fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])

#     # Plot all ROC curves
#     plt.figure()
#     for i in range(num_classes):
#         plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Multiclass ROC Curve')
#     plt.legend(loc="lower right")
#     plt.show()

# # Calculate the probabilities for each class
# model_predictions = []
# model_probabilities = []
# model_probabilities = F.softmax(torch.tensor(model_predictions), dim=0).numpy()

# plot_multiclass_roc_curve(y_test_binarized, model_probabilities, n_classes)

# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():  # Turn off gradients for the following block
#     for data, target in test_loader:
#         data, target = data.to(device), target.to(device)
#         output = model(data)
        
#         # Get class predictions
#         _, preds = torch.max(output, 1)
#         model_predictions.extend(preds.cpu().numpy())
        
#         # Get probabilities for the positive class
#         probs = F.softmax(output, dim=1)[:, 1]  # Adjust the index based on your positive class
#         model_probabilities.extend(probs.cpu().numpy())

# # # Specificity = 		    Number of true negatives (Number of true negatives + number of false positives) =		    
# # # = Total number of individuals without the illness

# # def sensitivity_specificity(conf_matrix):
# #     num_classes = conf_matrix.shape[0]
# #     sensitivity = np.zeros(num_classes)
# #     specificity = np.zeros(num_classes)

# #     for i in range(num_classes):
# #         TP = conf_matrix[i, i]
# #         FN = sum(conf_matrix[i, :]) - TP
# #         FP = sum(conf_matrix[:, i]) - TP
# #         TN = conf_matrix.sum() - (TP + FP + FN)

# #         sensitivity[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
# #         specificity[i] = TN / (TN + FP) if (TN + FP) != 0 else 0

# #     return sensitivity, specificity

# # from sklearn.preprocessing import label_binarize

# # # Binarize the labels for multiclass (suggestion of LLM)
# # y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))

# # #####################  R O C     C U R V E   #####################
# # def plot_multiclass_roc_curve(y_true, y_scores, num_classes):
# #     # Compute ROC curve and ROC area for each class
# #     fpr = dict()
# #     tpr = dict()
# #     roc_auc = dict()
    
# #     for i in range(num_classes):
# #         fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
# #         roc_auc[i] = auc(fpr[i], tpr[i])

# #     # Plot all ROC curves
# #     plt.figure()
# #     for i in range(num_classes):
# #         plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

# #     plt.plot([0, 1], [0, 1], 'k--')
# #     plt.xlim([0.0, 1.0])
# #     plt.ylim([0.0, 1.05])
# #     plt.xlabel('False Positive Rate')
# #     plt.ylabel('True Positive Rate')
# #     plt.title('Multiclass ROC Curve')
# #     plt.legend(loc="lower right")
# #     plt.show()

# # # Calculate the probabilities for each class
# # model_predictions = []
# # model_probabilities = []
# # model_probabilities = F.softmax(torch.tensor(model_predictions), dim=0).numpy()

# # plot_multiclass_roc_curve(y_test_binarized, model_probabilities, n_classes)

# # model.eval()  # Set the model to evaluation mode
# # with torch.no_grad():  # Turn off gradients for the following block
# #     for data, target in test_loader:
# #         data, target = data.to(device), target.to(device)
# #         output = model(data)
        
# #         # Get class predictions
# #         _, preds = torch.max(output, 1)
# #         model_predictions.extend(preds.cpu().numpy())
        
# #         # Get probabilities for the positive class
# #         probs = F.softmax(output, dim=1)[:, 1]  # Adjust the index based on your positive class
# #         model_probabilities.extend(probs.cpu().numpy())


# # # Calculate sensitivity and specificity
# # sensitivity, specificity = sensitivity_specificity(y_test, model_predictions)
# # print(f"Sensitivity: {sensitivity}")
# # print(f"Specificity: {specificity}")


# # ##################################################################################################################################################################

# # # # Display the images, 1 for each class
# # # def display_images(images, titles, num_images):
# # #     plt.figure(figsize=(15, 5))
# # #     for i in range(num_images):
# # #         image = np.squeeze(images[i]) # squeeze to make it easy to ptint in 2d
# # #         plt.subplot(1, num_images, i + 1)
# # #         plt.imshow(image, cmap='gray')
# # #         plt.title(titles[i])
# # #         plt.axis('off')
# # #     plt.show()

# # >>>>>>> ab59272 (Net / ResNet / EfficientNet  Experiments)
# # # data_train = np.load(data_train_path)


# # # # Data Verification to check if we all have everything good
# # # data_shape = data_train.shape
# # # data_type = data_train.dtype
# # # labels_shape = y_train.shape
# # # labels_type = y_train.dtype
# # # print(f"Data Shape: {data_shape}, Data Type: {data_type}")
# # # print(f"Labels Shape: {labels_shape}, Labels Type: {labels_type}")

# # # # Check the range and distribution of features
# # # data_range = (np.min(data_train), np.max(data_train))

# # # # Label Encoding in accordance to the diseases
# # # class_names_mapping = {
# # #     0: 'Atelectasis',
# # #     1: 'Effusion',
# # #     2: 'Infiltration',
# # #     3: 'No Finding',
# # #     4: 'Nodule',
# # #     5: 'Pneumonia'
# # # }
 
# # # print("Unique classes in the training set:")
# # # for class_id in unique_labels:
# # #     print(f"Class ID {class_id}: {class_names_mapping[class_id]}")

# # # # df for distribution analysis
# # # df_data_range = pd.DataFrame(data_train.reshape(data_train.shape[0], -1))


# # # Calculate the probabilities for each class
# # model_predictions = []
# # model_probabilities = []
# # model_probabilities = F.softmax(torch.tensor(model_predictions), dim=0).numpy()

# # plot_multiclass_roc_curve(y_test_binarized, model_probabilities, n_classes)

# # model.eval()  # Set the model to evaluation mode
# # with torch.no_grad():  # Turn off gradients for the following block
# #     for data, target in test_loader:
# #         data, target = data.to(device), target.to(device)
# #         output = model(data)
        
# #         # Get class predictions
# #         _, preds = torch.max(output, 1)
# #         model_predictions.extend(preds.cpu().numpy())
        
# #         # Get probabilities for the positive class
# #         probs = F.softmax(output, dim=1)[:, 1]  # Adjust the index based on your positive class
# #         model_probabilities.extend(probs.cpu().numpy())


# # # Calculate sensitivity and specificity
# # sensitivity, specificity = sensitivity_specificity(y_test, model_predictions)
# # print(f"Sensitivity: {sensitivity}")
# # print(f"Specificity: {specificity}")


# # ##################################################################################################################################################################

# # # # Display the images, 1 for each class
# # # def display_images(images, titles, num_images):
# # #     plt.figure(figsize=(15, 5))
# # #     for i in range(num_images):
# # #         image = np.squeeze(images[i]) # squeeze to make it easy to ptint in 2d
# # #         plt.subplot(1, num_images, i + 1)
# # #         plt.imshow(image, cmap='gray')
# # #         plt.title(titles[i])
# # #         plt.axis('off')
# # #     plt.show()

# # >>>>>>> ab59272 (Net / ResNet / EfficientNet  Experiments)
# # # data_train = np.load(data_train_path)


# # # # Data Verification to check if we all have everything good
# # # data_shape = data_train.shape
# # # data_type = data_train.dtype
# # # labels_shape = y_train.shape
# # # labels_type = y_train.dtype
# # # print(f"Data Shape: {data_shape}, Data Type: {data_type}")
# # # print(f"Labels Shape: {labels_shape}, Labels Type: {labels_type}")

# # # # Check the range and distribution of features
# # # data_range = (np.min(data_train), np.max(data_train))

# # # # Label Encoding in accordance to the diseases
# # # class_names_mapping = {
# # #     0: 'Atelectasis',
# # #     1: 'Effusion',
# # #     2: 'Infiltration',
# # #     3: 'No Finding',
# # #     4: 'Nodule',
# # #     5: 'Pneumonia'
# # # }
 
# # # print("Unique classes in the training set:")
# # # for class_id in unique_labels:
# # #     print(f"Class ID {class_id}: {class_names_mapping[class_id]}")

# # # # df for distribution analysis
# # # df_data_range = pd.DataFrame(data_train.reshape(data_train.shape[0], -1))


