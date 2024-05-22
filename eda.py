# # Imports
# import numpy as np
# import matplotlib.pyplot as plt
# import random

# maincolor = '#4a8cffff'
# secondcolor = '#e06666'

# NOTE: File used in the very beginning of the project. Please ignore!

# # Relative Path PUT YOUR PATHS HERE
# path = 'dc1/data/X_train.npy'

# data = np.load(path)


# # Display some images to see what are we working on
# def display_images(images, num_images=5):
#     plt.figure(figsize=(15, 3))
#     for i in range(num_images):
#         plt.subplot(1, num_images, i + 1)
#         plt.imshow(images[i].squeeze(), cmap='gray')
#         plt.axis('off')
#     plt.show()


# # function call
# display_images(data)


# ########################################################################################################################

# # # 1. Statistical Analysis
# # def compute_statistics(images):
# #     # flatten the images since x/y are irrelevant
# #     flattened_images = images.flatten()
# #     mean_val = np.mean(flattened_images)
# #     median_val = np.median(flattened_images)
# #     std_dev_val = np.std(flattened_images)

# #     return mean_val, median_val, std_dev_val


# # # Compute and print the statistics
# # mean_val, median_val, std_dev_val = compute_statistics(data)
# # print(f"Mean pixel intensity: {mean_val}")
# # print(f"Median pixel intensity: {median_val}")
# # print(f"Standard deviation of pixel intensities: {std_dev_val}")

# # # Global statistics
# # global_mean = np.mean(data)
# # global_std = np.std(data)
# # # Individual image statistics
# # image_means = np.mean(data, axis=(1, 2, 3))
# # image_stds = np.std(data, axis=(1, 2, 3))
# # # Outlier thresholds 
# # upper_threshold = global_mean + 3 * global_std
# # lower_threshold = global_mean - 3 * global_std
# # outlier_indices = np.where((image_means > upper_threshold) | (image_means < lower_threshold))[0]
# # print(f"Found {len(outlier_indices)} potential outliers based on pixel intensity means.")


# # ########################################################################################################################

# # # 2. Histogram Analysis
# # def plot_histogram(images, title="Pixel Intensity Distribution"):
# #     flattened_images = images.flatten()

# #     # Customize plot aesthetics
# #     plt.figure(figsize=(10, 6))
# #     plt.hist(flattened_images, bins=256, range=(0, 255), color= maincolor, alpha=0.75)
    
# #     # Adding grid, title, and labels with improved aesthetics
# #     plt.grid(axis='y', alpha=0.75)
# #     plt.title(title, fontsize=15, color='#333333')
# #     plt.xlabel('Pixel Intensity', fontsize=12, color='#333333')
# #     plt.ylabel('Frequency', fontsize=12, color='#333333')

# #     # Customizing tick marks for better readability
# #     plt.xticks(fontsize=10, color='#333333')
# #     plt.yticks(fontsize=10, color='#333333')

# #     # Adding a background color to the plot for contrast
# #     ax = plt.gca()  # Get current axes
# #     ax.set_facecolor('#f0f0f0')
# #     ax.figure.set_facecolor('#f8f8f8')

# #     # Add a border around the plot for a more polished look
# #     for spine in ax.spines.values():
# #         spine.set_edgecolor('#d0d0d0')

# #     plt.show()

# # # Plot histogram for the entire dataset
# # plot_histogram(data, title="Pixel Intensity Distribution Across Entire Dataset")
# # # Plot a selected image
# # plot_histogram(data[10], title="Pixel Intensity Distribution of a Selected Image")


# def plot_histogram_with_images(images, num_images=5):
#     # Select a set of random images
#     random_indices = random.sample(range(images.shape[0]), num_images)

#     for index in random_indices:
#         # Extract a single image
#         single_xray_image = images[index]

#         # Flatten the image for histogram
#         flattened_image = single_xray_image.flatten()

#         # Create a figure with 2 subplots
#         fig, axs = plt.subplots(1, 2, figsize=(12, 6))

#         # Plot histogram on the first subplot
#         axs[0].hist(flattened_image, bins=256, range=(0, 255), color=maincolor, alpha=0.75)
#         axs[0].set_title('Pixel Intensity Distribution')
#         axs[0].set_xlabel('Pixel Intensity')
#         axs[0].set_ylabel('Frequency')
#         axs[0].set_ylim(0,600)
#         axs[0].grid(True)

#         # Show the image on the second subplot
#         axs[1].imshow(single_xray_image.squeeze(), cmap='gray')
#         axs[1].set_title('X-Ray Image')
#         axs[1].axis('off')

#     plt.tight_layout()
#     plt.show()

# plot_histogram_with_images(data)


# # 3. Plot for Accuracy, Precision and # Recall
# def plot_metrics_evolution(epochs, accuracy, precision):
#     import matplotlib.pyplot as plt

#     plt.rcParams.update({'font.size': 12})

#     plt.figure(figsize=(12, 8))

#     plt.plot(epochs, accuracy, label='Accuracy', marker='o', linestyle='-', color=maincolor)
#     plt.plot(epochs, precision, label='Precision', marker='s', linestyle='--', color=secondcolor)
# #    plt.plot(epochs, recall, label='Recall', marker='^', linestyle='-.', color='red')

#     plt.title('Model Performance Over 10 Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Score')
#     plt.xticks(epochs)

#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Data
# epochs = list(range(1, 11))
# accuracy = [0.1884, 0.1968, 0.1985, 0.2200, 0.2122, 0.2208, 0.2340, 0.2337, 0.2318, 0.2384]
# precision = [0.1664, 0.3518, 0.2644, 0.3144, 0.3137, 0.3212, 0.2983, 0.3108, 0.2635, 0.3081]
# # recall = [0.1884, 0.1968, 0.1985, 0.2200, 0.2122, 0.2208, 0.2340, 0.2337, 0.2318, 0.2384]

# # Example function call
# plot_metrics_evolution(epochs, accuracy, precision)