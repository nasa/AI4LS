import numpy as np
import os
from skimage.measure import find_contours
import matplotlib.pyplot as plt

class Visualize_GradCAM_Features:
    def __init__(self, image_path, input_image, heatmap, save_path):
        self.image_path = image_path
        self.input_image = input_image
        self.heatmap = heatmap
        self.save_path = save_path

    def visualize_clusters_on_heatmap(self, binary_map):
        """
        Visualize clusters on the Grad-CAM heatmap by drawing contours around regions 
        where the heatmap value exceeds a threshold.
        
        Parameters:
            image_path (str): Image path
            input_image: Image in PIL Format
            heatmap (np.array): 2D Grad-CAM heatmap (normalized to 0-255 or similar scale).
        """
        # Extract the filename
        filename = os.path.basename(self.image_path)
        # Extract the final folder name
        final_foldername = os.path.basename(os.path.dirname(self.image_path))
        
        # Find contours in the binary map at the 0.5 level
        contours = find_contours(binary_map, level=0.5)
        
        # Plot the heatmap and overlay the contours
        plt.figure(figsize=(8, 8))
        plt.imshow(self.input_image, cmap='gray')
        plt.imshow(self.heatmap, cmap='jet', alpha=0.35)
        for contour in contours:
            # contour coordinates: (row, col)
            plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='white')
        
        plt.title(f"Cluster Contours\n{os.path.join(final_foldername, filename)}")
        plt.axis('off')
        # plt.show()

        # Save the image
        save_path = os.path.join(self.save_path, "gradcam_clusters", os.path.basename(os.path.dirname(self.image_path)), os.path.splitext(os.path.basename(self.image_path))[0]+".png")
        print("Saving", save_path)
        # Create the target directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    def visualize_com_on_heatmap(self, com_x, com_y):
        """
        Visualize Center of Mass (CoM) on the Grad-CAM heatmap.
        
        Parameters:
            image_path (str): Image path
            input_image: Image in PIL Format
            com_x, com_y: X and Y co-ordinates for the COM.
            heatmap (np.array): 2D Grad-CAM heatmap (normalized to 0-255 or similar scale).
        """

        # Plot the Grad-CAM heatmap with Center of Mass
        plt.figure(figsize=(8, 8))
        plt.imshow(self.input_image, cmap='gray')  # Original grayscale image
        plt.imshow(self.heatmap, cmap='jet', alpha=0.35)  # Heatmap overlay
        plt.scatter(com_x, com_y, color='red', marker='x', s=75, label='Center of Mass')  # Plot CoM
        plt.title("Grad-CAM Heatmap with Center of Mass")
        plt.legend()
        plt.axis('off')

        # plt.show()

        # Save the image
        save_path = os.path.join(self.save_path, "gradcam_CoM", os.path.basename(os.path.dirname(self.image_path)), os.path.splitext(os.path.basename(self.image_path))[0]+".png")
        print("Saving", save_path)
        # Create the target directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()