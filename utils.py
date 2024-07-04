import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops, regionprops_table

def plots_results_on_image(input_image, labels_image, results_array, title):
    """Plots results array on top of each label"""
    plt.figure(figsize=(10, 10))
    plt.imshow(input_image, cmap='viridis')
    
    for label in range(1, np.max(labels_image) + 1):
        positions = np.column_stack(np.where(labels_image == label))
        if len(positions) > 0:
            # Choose the first position found for each label to display the percentage
            y, x = positions[0]
            plt.text(x, y - 25, f'{results_array[label - 1]:.2f}', color='white', fontsize=8, ha='center')

    plt.title(title)
    plt.axis('off')
    plt.show()

def extract_labels(img, threshold_ch0, threshold_ch1):

    # Extract channel array
    ch_0 = img[0,:,:]
    ch_1 = img[1,:,:]
    # Create a binary mask using simple thresholding
    ch_0_mask = ch_0 > threshold_ch0
    ch_1_mask = ch_1 > threshold_ch1
    # Transform mask into objects using connected component analysis
    ch_0_labels = label(ch_0_mask)
    ch_1_labels = label(ch_1_mask)

    return ch_0, ch_1, ch_0_labels, ch_1_labels


def calculate_colocalization(ch_0_labels, ch_1_labels):
    """"""
    # Ensure that the input arrays have the same shape
    assert ch_0_labels.shape == ch_1_labels.shape, "Input arrays must have the same shape"

    # Ensure that both channels have the same number of labels
    assert np.max(ch_0_labels) == np.max(ch_1_labels), "Input arrays must have the same number of labels"

    # Define the max number of labels
    label_nr = np.max(ch_0_labels)

    # Create an array that stores a True value for each of the xy positions where both ch_0 and ch_1 labels are present
    positive_both = (ch_0_labels > 0) & (ch_1_labels > 0)

    # Count positive pixels for each label in ch_0

    # Initialize an empty array to store results 
    label_counts = np.zeros(label_nr) 

    # Loop over each label
    for label in range(1, (label_nr + 1)):
        # Update the label_counts array with the sum of pixels for each ch0 labelled pixel that is also present in ch1 (positive both)
        label_counts[label - 1] = np.sum((ch_0_labels == label) & positive_both)

    # Calculate the total number of pixels for each label in ch_0_labels
    total_label_counts = np.array([np.sum(ch_0_labels == label) for label in range(1, (label_nr + 1))])
    # Calculate the percentage of positive pixels for both labels
    percentage_positive_both = (label_counts / total_label_counts) * 100
    
    return percentage_positive_both


def calculate_props(label_image, intensity_image):
    """Extract morphology and mean intensity from each label, returns an array of results"""
    results_array = regionprops_table(label_image ,intensity_image, properties=['label','area_filled','area','intensity_mean'])

    return results_array