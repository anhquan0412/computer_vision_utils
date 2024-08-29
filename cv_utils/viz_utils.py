import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def clas_report_compact(y_true,y_pred,label_names=None):
    report = classification_report(y_true, y_pred, 
                               target_names=label_names, 
                               output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df_short = report_df[report_df.support>0]
    report_df_short = report_df_short.iloc[:-3].copy()
    report_df_short.support = report_df_short.support.astype(int)
    return report_df_short

def plot_classification_report(report_df_short,rotation=85,figsize=(10,8),fontsize=8,metrics=['f1','precision','recall']):
    species = report_df_short.index.values
    support = report_df_short['support'].values
    f1_score = report_df_short['f1-score'].values
    precision = report_df_short['precision'].values
    recall = report_df_short['recall'].values

    # Sort by support in descending order
    _idxs = np.argsort(support)[::-1]
    species = species[_idxs]
    support = support[_idxs]
    f1_score = f1_score[_idxs]
    precision = precision[_idxs]
    recall = recall[_idxs]

    fig, ax1 = plt.subplots(figsize=figsize)

    # Plotting support numbers
    color = 'tab:blue'
    ax1.set_xlabel('Species')
    ax1.set_ylabel('Support', color=color)
    ax1.bar(species, support, color=color, alpha=0.6, label='Support')
    ax1.tick_params(axis='y', labelcolor=color)

    # Rotate species labels and set font size
    plt.xticks(rotation=rotation, fontsize=fontsize)

    # Creating a second y-axis to plot f1-scores, precision, and recall
    ax2 = ax1.twinx()
    ax2.set_ylabel('Scores', color='black')
    if 'f1' in metrics:
        ax2.plot(species, f1_score, color='tab:red', marker='o', linestyle='-', linewidth=2, label='F1-Score')
    if 'precision' in metrics:
        ax2.plot(species, precision, color='tab:green', marker='s', linestyle='--', linewidth=2, label='Precision')
    if 'recall' in metrics:
        ax2.plot(species, recall, color='tab:purple', marker='^', linestyle='-.', linewidth=2, label='Recall')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 1.1)  # Set y-axis to start at 0

    fig.suptitle('Support And Performance By Species')
    fig.tight_layout()
    fig.legend()
    
    ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    ax2.minorticks_on()
    ax2.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.5)
    plt.show()

def plot_confusion_matrix(cm,label_names=None,fontsize=8,figsize=(12, 12)):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=figsize)
    disp.plot(ax=ax, values_format='d')

    for text in disp.text_.ravel():
        text.set_fontsize(fontsize)  # Set the desired font size
    plt.xticks(rotation=90, fontsize=10)
    plt.show()


def visualize_images(image_paths, labels=None, figsize=(10, 10), fontsize=8):
    # Determine grid size based on the number of images
    num_images = len(image_paths)
    grid_size = math.ceil(math.sqrt(num_images))

    if labels is not None and not isinstance(labels,list):
        labels = [labels for _ in range(num_images)]
    # Create the plot with specified figure size
    fig, axs = plt.subplots(grid_size, grid_size, figsize=figsize)
    
    # If only one image, axs is not a 2D array, so we handle it separately
    if num_images == 1:
        axs = [axs]
    else:
        axs = axs.ravel()
    
    # Iterate through the images and plot them
    for i in range(num_images):
        img = mpimg.imread(image_paths[i])
        axs[i].imshow(img)
        if labels is not None:
            axs[i].set_title(f'{labels[i]}', fontsize=fontsize)
        axs[i].axis('off')
    
    # Hide any unused subplots if the grid is larger than the number of images
    for j in range(num_images, len(axs)):
        axs[j].axis('off')
    
    # Display the plot
    plt.tight_layout()
    plt.show()