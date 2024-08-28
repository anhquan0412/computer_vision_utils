import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def visualize_images(image_paths, labels, figsize=(10, 10), fontsize=8):
    # Determine grid size based on the number of images
    num_images = len(image_paths)
    grid_size = math.ceil(math.sqrt(num_images))

    if not isinstance(labels,list):
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
        axs[i].set_title(f'{labels[i]}', fontsize=fontsize)
        axs[i].axis('off')
    
    # Hide any unused subplots if the grid is larger than the number of images
    for j in range(num_images, len(axs)):
        axs[j].axis('off')
    
    # Display the plot
    plt.tight_layout()
    plt.show()