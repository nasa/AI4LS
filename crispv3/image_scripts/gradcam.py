import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from image_scripts.image_classifier import TransferLearningImageClassifier, CNN_Scratch 
from torchvision import transforms
import pandas as pd

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Hook to capture forward pass activations
        self.target_layer.register_forward_hook(self.save_activation)
        # Hook to capture backward pass gradients
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Save forward pass activations."""
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients."""
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor):
        # Forward pass
        output = self.model(input_tensor)  # Raw logits
        probabilities = torch.sigmoid(output)  # Convert logits to probabilities
        predicted_class = (probabilities > 0.5).float().item()  # Binary prediction (0 or 1)

        # Backward pass for the predicted class
        self.model.zero_grad()
        if output.shape[1] == 1:  # Binary classification with one logit
            target = output  # Use the raw logit for backward pass
        else:  # Binary classification with two logits
            target = output[:, predicted_class]  # Use the predicted class logit
        target.backward(retain_graph=True)

        # Get activations and gradients
        activations = self.activations
        gradients = self.gradients

        # Compute Grad-CAM weights
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # Global average pooling
        weighted_activations = (weights * activations).sum(dim=1)  # Weighted sum

        # Normalize the heatmap
        heatmap = torch.relu(weighted_activations)
        heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize to [0, 1]

        # Resize heatmap to match input image
        heatmap = F.interpolate(heatmap.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

        return heatmap.detach().cpu().squeeze().numpy(), predicted_class

def save_gradcam_output(config, model):
    """Generate and save Grad-CAM heatmaps for the full dataset."""
    # Extract paths and parameters
    image_dir = config["image_data"]["image_dir"]
    labels_csv = config["image_data"]["labels_csv"]
    model_save_path = config["image_data"].get("model_save_path", os.path.join("image_model_saved", "image_model.pth"))
    model_type = config["image_data"].get("model_type", "DenseNet121")  # Default to DenseNet121
    gradcam_output_save_path = config["image_data"]["image_model_gradcam"]["gradcam_output_save_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select the correct target convolutional layer for Grad-CAM
    if model_type == "DenseNet121":  # DenseNet121
        model = TransferLearningImageClassifier().to(device)
        target_layer = model.model.features[-2]
    else:  # CNN_Scratch
        model = CNN_Scratch().to(device)
        target_layer = model.conv4

    checkpoint = torch.load(model_save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode

    grad_cam = GradCAM(model, target_layer)  # Initialize Grad-CAM once

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    labels_df = pd.read_csv(labels_csv)
    os.makedirs(gradcam_output_save_path, exist_ok=True)

    # Loop through test images in the dataset
    for index, row in labels_df.iterrows():
        # Load the test image
        input_image = np.load(os.path.join(image_dir, row['image_name']))
        # Convert to PIL Image for transformations
        input_image = (input_image * 255).astype(np.uint8)  # Scale back to [0, 255]
        input_image = Image.fromarray(input_image)

        # Apply transformations
        input_tensor = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension

        # Generate the heatmap and predicted class
        print("\nGenerating heatmap for", row['image_name'])
        heatmap, predicted_class = grad_cam.generate_heatmap(input_tensor)

        # Overlay Grad-CAM heatmap
        plt.imshow(input_image, cmap='gray')  # Original grayscale image
        plt.imshow(heatmap, alpha=0.35, cmap='jet')  # Grad-CAM overlay
        plt.title(f"{row['image_name'].split('.')[0]} \n(Predicted: {predicted_class})")
        plt.colorbar()

        # Save the image
        save_path, _ = os.path.splitext(os.path.join(gradcam_output_save_path, row['image_name']))  # Splits path
        # Create the target directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Grad-CAM to {save_path}")