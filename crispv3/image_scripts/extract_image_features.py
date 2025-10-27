import torch
import numpy as np
import pandas as pd
import os
import cv2
from torchvision import transforms
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from scipy.ndimage import center_of_mass
from scipy.stats import moment, kurtosis, skew
from image_scripts.image_classifier import CNN_Scratch, TransferLearningImageClassifier
from image_scripts.gradcam import GradCAM
from image_scripts.visualize_gradcam_features import Visualize_GradCAM_Features
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def load_image(image_path):
        """Loads an image from file (.png, .jpg) or `.npy` array (NumPy)."""
        if image_path.endswith(".npy"):
            image = np.load(image_path)  # Load NumPy array
            if image.ndim != 2:  # Ensure it's grayscale
                raise ValueError(f"Invalid shape {image.shape} for grayscale image {image_path}")
            return Image.fromarray((image * 255).astype(np.uint8), mode="L")  # Convert to PIL grayscale
        else:
            return Image.open(image_path).convert("L")  # Load regular image file
        
def compute_radial_features(heatmap, bin_width=5):
    """
    Compute radial features for a heatmap by grouping pixel distances into bins.
    
    Parameters:
        heatmap (np.array): 2D array (grayscale) representing the Grad-CAM heatmap.
        bin_width (int): Width of each radial bin in pixels.
    
    Returns:
      radial_features (list): List of mean activation values for each radial bin.
    """
    # Determine the center of the heatmap
    center = np.array(heatmap.shape) // 2
    
    # Compute distances from the center for each pixel
    indices = np.indices(heatmap.shape)
    distances = np.sqrt((indices[0] - center[0])**2 + (indices[1] - center[1])**2)
    
    max_distance = int(np.max(distances))
    num_bins = max_distance // bin_width
    radial_features = []
    
    for b in range(num_bins):
        lower = b * bin_width + 1
        upper = (b + 1) * bin_width
        # Select pixels within the current bin
        bin_pixels = heatmap[(distances >= lower) & (distances < upper)]
        # Compute the mean activation; replace NaN with 0.0 if needed
        bin_mean = np.nan_to_num(np.mean(bin_pixels), nan=0.0) if bin_pixels.size > 0 else 0.0
        radial_features.append(bin_mean)
    
    return radial_features

def extract_gradcam_features(config, model, image_path, target_layer, visualize=False):
    """ Generate Grad-CAM heatmap and extract structured numerical features. """

    show_clusters = config["image_data"]["gradcam_features_explainer"].get("show_clusters", False)
    show_com = config["image_data"]["gradcam_features_explainer"].get("show_com", False)
    gradcam_features_explainer_save_path = config["image_data"]["gradcam_features_explainer"].get("save_path", "gradcam_features_explainer")

    # Load and preprocess image
    input_image = load_image(image_path)  # Load correctly based on file type

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = transform(input_image).unsqueeze(0).to(device)

    # Generate Grad-CAM heatmap
    grad_cam = GradCAM(model, target_layer)
    original_heatmap, predicted_class = grad_cam.generate_heatmap(input_tensor)

    # Ensure heatmap is in the correct range
    heatmap = np.nan_to_num(original_heatmap, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaNs/Infs
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute basic activation statistics
    mean_activation = np.mean(heatmap)
    max_activation = np.max(heatmap)
    spread_variance = np.var(heatmap)

    # Compute Center of Mass
    com_y, com_x = center_of_mass(heatmap)  # Returns (y, x) coordinates

    radial_features = compute_radial_features(heatmap)

    # Compute texture-based features using Gray-Level Co-Occurrence Matrix (GLCM)
    glcm = graycomatrix(heatmap, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Compute topological features: number of clusters, average cluster size
    binary_map = (heatmap > (np.mean(heatmap) + np.std(heatmap))).astype(np.uint8)
    labeled_clusters = label(binary_map)
    cluster_props = regionprops(labeled_clusters)
    num_activation_clusters = len(cluster_props)
    avg_cluster_size = np.mean([prop.area for prop in cluster_props]) if cluster_props else 0
    # Normalize avg_cluster_size
    heatmap_size = heatmap.shape[0] * heatmap.shape[1]

    if(show_clusters or show_com):
        feature_visualizer = Visualize_GradCAM_Features(image_path, input_image, heatmap, gradcam_features_explainer_save_path)
        if(show_clusters):
            print("Saving Clusters on gradcam visualizations . . .")
            feature_visualizer.visualize_clusters_on_heatmap(binary_map)
        if(show_com):
            print("Saving Center of Mass pixels on gradcam visualizations . . .")
            feature_visualizer.visualize_com_on_heatmap(com_x, com_y)

    # Compute spatial connectivity of clusters
    cluster_connectivity = np.sum(binary_map * np.roll(binary_map, 1, axis=0) * np.roll(binary_map, 1, axis=1))
    # Normalize cluster_connectivity
    max_connections = heatmap_size * 4  # Assuming 4-connected neighbors
    cluster_connectivity_normalized = cluster_connectivity / max_connections

    # Compute normalized moments
    heatmap_flat = heatmap.flatten()

    # Additional moments (more meaningful)
    normalized_moment_skewness  = skew(heatmap_flat)
    normalized_moment_kurtosis = kurtosis(heatmap_flat)

    # Aggregate all features
    features = [
        mean_activation, max_activation, spread_variance,
        contrast, homogeneity, energy, correlation,
        num_activation_clusters, avg_cluster_size, cluster_connectivity_normalized, com_x, com_y, normalized_moment_skewness, normalized_moment_kurtosis
    ] + radial_features

    return features, radial_features

def extract_image_features(config, model):
    """ Extract CNN feature embeddings and Grad-CAM features from images """
    
    # Extract paths and parameters from config
    subject_key = config["data_options"]["subject_keys"]
    target_var = config["data_options"]["targets"][0]
    environments = config["data_options"]["environments"][0]

    # model_save_path = config["image_data"]["model_save_path"]
    model_save_path = config["image_data"].get("model_save_path", os.path.join("image_model_saved", "image_model.pth"))
    image_dir = config["image_data"]["image_dir"]
    labels_csv = config["image_data"]["labels_csv"]
    gradcam_features_save_path = config["image_data"].get("gradcam_features_save_path", os.path.dirname(image_dir))
    model_type = config["image_data"].get("model_type", "DenseNet121")

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransferLearningImageClassifier().to(device) if model_type == "DenseNet121" else CNN_Scratch().to(device)

    checkpoint = torch.load(model_save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = model.model.features[-2] if model_type == "DenseNet121" else model.conv4

    # Read dataset CSV
    df = pd.read_csv(labels_csv)
    gradcam_feature_list = []

    for _, row in df.iterrows():
        sample = row[subject_key]
        img_filename = row['image_name']
        label = row[target_var]
        env_split = row[environments]
        img_path = os.path.join(image_dir, img_filename)

        if os.path.exists(img_path):
            # Extract Grad-CAM Features
            gradcam_features, radial_features = extract_gradcam_features(config, model, img_path, target_layer, visualize=False)

            # Store Features
            gradcam_feature_list.append([sample, img_filename, label, env_split] + gradcam_features)
        else:
            print(f"Warning: Image {img_filename} not found. Skipping.")

    # Define meaningful column names
    feature_names = [
        "mean_activation", "max_activation", "spread_variance",
        "contrast", "homogeneity", "energy", "correlation",
        "num_activation_clusters", "avg_cluster_size", "cluster_connectivity_normalized", 
        "center_of_mass_x", "center_of_mass_y", 
        "normalized_moment_skewness", "normalized_moment_kurtosis"
    ]

    # Example: Bin width = 5
    bin_width = 5
    radial_feature_names = [
        f"radial_mean_activation_{i * bin_width + 1}_{(i + 1) * bin_width}px"
        for i in range(len(radial_features))
    ]

    # Combine all feature names
    gradcam_column_names = [subject_key, "image_name", target_var, environments] + feature_names + radial_feature_names

    # Verify feature lengths match
    for row in gradcam_feature_list:
        assert len(row) == len(gradcam_column_names), f"Feature length mismatch: Expected {len(gradcam_column_names)}, got {len(row)}"

    # Save Grad-CAM features
    gradcam_features_df = pd.DataFrame(gradcam_feature_list, columns=gradcam_column_names)
    gradcam_features_df[feature_names + radial_feature_names] = gradcam_features_df[feature_names + radial_feature_names].fillna(0)

    # Normalize the remaining columns using MinMaxScaler
    print('Normalizing image-gradcam features . . .')
    scaler = MinMaxScaler()  # Use StandardScaler() for standardization instead
    features_to_normalize=feature_names + radial_feature_names
    features_to_normalize.remove('num_activation_clusters')
    gradcam_features_df[features_to_normalize] = scaler.fit_transform(gradcam_features_df[features_to_normalize])

    gradcam_features_df.to_csv(gradcam_features_save_path.split('.')[0]+'.csv', index=False)
    gradcam_features_df.reset_index(drop=True).to_pickle(gradcam_features_save_path)
    print(f"Extracted Grad-CAM features saved to {gradcam_features_save_path}")
