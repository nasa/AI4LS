import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
import os
import random
from image_scripts.image_classifier import TransferLearningImageClassifier, CNN_Scratch 
import json
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

class ImageDataset(Dataset):
    """ Custom Dataset for Grayscale Image Classification """

    def __init__(self, image_dir, target_label, labels_csv, transform=None, augment=False):
        """
        Args:
            image_dir (str): Path to the directory containing images (.png, .jpg, or .npy).
            target_label (str): Column name of the target variable in the labels_csv
            labels_csv (str): Path to CSV file with image filenames and labels.
            transform (callable, optional): Transform to be applied on an image.
            augment (bool): Whether to apply data augmentation.
        """

        self.image_dir = image_dir
        self.target_label=target_label
        self.data = pd.read_csv(labels_csv)
        self.image_shape = (224, 224)
        
        # Standard transform for both train and validation sets
        base_transform = transforms.Compose([
            transforms.Resize(self.image_shape),  # Resize for model
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Augmentations applied only to training data
        if augment:
            print("Applying training image augmentation . . .")
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=10),
                transforms.RandomResizedCrop(size=self.image_shape, scale=(0.9, 1.0)),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                *base_transform.transforms  # Append base transforms
            ])
        else:
            self.transform = base_transform  # Use only base transform

    def __len__(self):
        return len(self.data)

    def load_image(self, image_path):
        """Loads an image from file (.png, .jpg) or `.npy` array (NumPy)."""
        if image_path.endswith(".npy"):
            image = np.load(image_path)  # Load NumPy array
            if image.ndim != 2:  # Ensure it's grayscale
                raise ValueError(f"Invalid shape {image.shape} for grayscale image {image_path}")
            return Image.fromarray((image * 255).astype(np.uint8), mode="L")  # Convert to PIL grayscale
        else:
            return Image.open(image_path).convert("L")  # Load regular image file

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx]['image_name']) # Get image filename
        label = torch.tensor(self.data.loc[idx, self.target_label], dtype=torch.float32)  # Get label

        image = self.load_image(img_name)  # Load image
        image = self.transform(image)  # Apply transforms

        return image, label

def set_global_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    

def save_model(model, optimizer, model_save_path, save_optimizer=False):
    """ Saves the trained model and optimizer state (if required). """

    # Extract the directory path from the save path
    model_dir = os.path.dirname(model_save_path)

    # Ensure the directory exists
    if not os.path.exists(model_dir):
        print(f"Warning: Directory '{model_dir}' does not exist. Creating it now...")
        os.makedirs(model_dir)  # Create directory

    # Save the model (with or without optimizer)
    save_dict = {"model_state_dict": model.state_dict()}
    if save_optimizer:
        save_dict["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(save_dict, model_save_path)
    print(f"Model saved successfully to {model_save_path}")


def train_image_model(config):
    """ Train an image classification model based on config settings """

    # If config is a dictionary, use it directly. Otherwise, load from file.
    if isinstance(config, dict):
        pass  # Use the given dictionary
    else:
        with open(config, "r") as f:
            config = json.load(f)
    
    # Extract paths and parameters
    target_label = config['data_options']['targets'][0]
    image_dir = config["image_data"]["image_dir"]
    labels_csv = config["image_data"]["labels_csv"]
    model_type = config["image_data"].get("model_type", "DenseNet121")
    batch_size = config["image_data"].get("batch_size", 16)
    learning_rate = config["image_data"].get("learning_rate", 0.0001)
    num_epochs = config["image_data"].get("num_epochs", 100)
    model_save_path = config["image_data"].get("model_save_path", os.path.join("image_model_saved", "image_model.pth"))
    split_ratio = config["image_data"].get("split_ratio", 0.8)  # Default: 80% train, 20% val
    augmentation = config["image_data"].get("augmentation", False)
    seed = config["feature_selection_options"].get("seed", 123)

    # Set Seed
    set_global_seed(seed)

    # Load dataset
    full_dataset = ImageDataset(image_dir, target_label, labels_csv)

    # Split dataset into train and validation
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Set the seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Wrap train dataset with augmentations
    train_dataset.dataset = ImageDataset(image_dir, target_label, labels_csv, augment=augmentation)

    # Create DataLoaders
    full_dataset_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose correct model architecture
    if model_type == "DenseNet121":
        print(f"Running image model {model_type}")
        model = TransferLearningImageClassifier().to(device)
    else:
        print(f"Running image model {model_type}")
        model = CNN_Scratch().to(device)

    # Compute class weights
    train_labels_list = [label for _, label in train_dataset]
    train_labels_array = np.array(train_labels_list)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels_array),
        y=train_labels_array
    )
    pos_weight = torch.tensor([class_weights[1]], dtype=torch.float).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_losses = []
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation loop
        val_losses = []
        correct = 0
        total = 0
        all_labels = []
        all_probs = []
        all_preds = []

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()

                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        accuracy = correct / total

        try:
            auc_score = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc_score = None

        try:
            f1 = f1_score(all_labels, all_preds)
        except ValueError:
            f1 = None

        if config.get("verbose", 1) == 1:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  Val Accuracy: {accuracy:.4f}")
            if auc_score is not None:
                print(f"  Val AUC: {auc_score:.4f}")
            else:
                print("  Val AUC: Not Computable (single class)")
            if f1 is not None:
                print(f"  Val F1-Score: {f1:.4f}")
            else:
                print("  Val F1-Score: Not Computable (single class)")


    # Save the trained model
    save_model(model, optimizer, model_save_path)

    print(f"Final model {model_type} trained on the train data has been saved.")

    return model

def train_image_model_loocv(config):
    """
    Train an image classification model using Leave-One-Out Cross Validation (LOOCV).
    
    For each sample in the dataset, the model is trained on all other samples and evaluated
    on the held-out sample. Performance metrics are printed for each fold. After LOOCV is completed,
    the model is retrained on the full dataset and saved to disk.

    Args:
        config (dict or str): Configuration dictionary or path to a JSON file.
    """

    # If config is a dictionary, use it directly. Otherwise, load from file.
    if isinstance(config, dict):
        pass  # Use the given dictionary
    else:
        with open(config, "r") as f:
            config = json.load(f)
    
    # Extract parameters and paths
    target_label = config['data_options']['targets'][0]
    image_dir = config["image_data"]["image_dir"]
    labels_csv = config["image_data"]["labels_csv"]
    model_type = config["image_data"].get("model_type", "DenseNet121")
    batch_size = config["image_data"].get("batch_size", 16)
    learning_rate = config["image_data"].get("learning_rate", 0.0001)
    num_epochs = config["image_data"].get("num_epochs", 100)
    augmentation = config["image_data"].get("augmentation", False)
    model_save_path = config["image_data"].get("model_save_path", os.path.join("image_model_saved", "image_model.pth"))
    verbose = config.get("verbose", 0)
    seed = config["feature_selection_options"].get("seed", 123)

    # Set Seed
    set_global_seed(seed)
    
    # Load CSV to determine dataset size.
    labels_df = pd.read_csv(labels_csv)
    total_samples = len(labels_df)
    
    # Create datasets: training with augmentation and validation without.
    train_dataset_full = ImageDataset(image_dir, target_label, labels_csv, augment=augmentation)
    val_dataset_full = ImageDataset(image_dir, target_label, labels_csv, augment=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fold_accuracies = []
    all_true_labels = []
    all_pred_probs = []
    
    # LOOCV: iterate over each sample as the held-out validation set.
    for i in range(total_samples):
        if verbose:
            print(f"Starting fold {i+1}/{total_samples}")
        # All indices except the i-th sample for training; the i-th for validation.
        train_indices = list(range(total_samples))
        train_indices.remove(i)
        val_indices = [i]
        
        train_subset = Subset(train_dataset_full, train_indices)
        val_subset = Subset(val_dataset_full, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
        
        # Initialize the model for this fold.
        if model_type == "DenseNet121":
            model = TransferLearningImageClassifier().to(device)
        else:
            model = CNN_Scratch().to(device)
        
        # Compute class weights based on training subset.
        train_labels_list = [label.item() for _, label in train_subset]
        train_labels_array = np.array(train_labels_list)
        classes = np.unique(train_labels_array)
        if len(classes) > 1:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=train_labels_array
            )
            pos_weight = torch.tensor([class_weights[1]], dtype=torch.float).to(device)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop for the current fold.
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate on the held-out sample.
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                prob = torch.sigmoid(outputs)
                prediction = (prob > 0.5).float()
                correct = (prediction == labels).item()
                fold_accuracies.append(correct)
                all_true_labels.append(labels.item())
                all_pred_probs.append(prob.item())
                if verbose:
                    print(f"Fold {i+1}: True Label: {labels.item()}, Predicted Prob: {prob.item()}, Accuracy: {correct}")
    
    # Print aggregated LOOCV metrics.
    avg_accuracy = np.mean(fold_accuracies)
    try:
        auc_score = roc_auc_score(all_true_labels, all_pred_probs)
    except ValueError:
        auc_score = None
    print(f"LOOCV Average Accuracy: {avg_accuracy:.4f}")
    if auc_score is not None:
        print(f"LOOCV AUC Score: {auc_score:.4f}")
    else:
        print("LOOCV AUC Score could not be computed (only one class present across folds).")
    
    # --- Retrain Final Model on Full Dataset and Save ---
    print("Training final model on full dataset...")
    # Use the training dataset (with augmentation if enabled).
    full_loader = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True)
    
    if model_type == "DenseNet121":
        final_model = TransferLearningImageClassifier().to(device)
    else:
        final_model = CNN_Scratch().to(device)
    
    # Compute class weights for the full dataset.
    all_labels_list = [label.item() for _, label in train_dataset_full]
    all_labels_array = np.array(all_labels_list)
    classes = np.unique(all_labels_array)
    if len(classes) > 1:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=all_labels_array
        )
        pos_weight = torch.tensor([class_weights[1]], dtype=torch.float).to(device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        final_model.train()
        for images, labels in full_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = final_model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Final model training - Epoch {epoch+1}/{num_epochs} completed.")
    
    # Save the final model.
    save_model(final_model, optimizer, model_save_path)
    print(f"Final model saved to {model_save_path}")

    return final_model

def train_image_model_kfold(config):
    # Parameters
    target_label = config['data_options']['targets'][0]
    k_folds = config["image_data"].get("k_folds", 3)
    model_type = config["image_data"].get("model_type", "DenseNet121")
    batch_size = config["image_data"].get("batch_size", 16)
    learning_rate = config["image_data"].get("learning_rate", 0.0001)
    num_epochs = config["image_data"].get("num_epochs", 100)
    augmentation = config["image_data"].get("augmentation", False)
    image_dir = config["image_data"]["image_dir"]
    labels_csv = config["image_data"]["labels_csv"]
    model_save_path = config["image_data"].get("model_save_path", os.path.join("image_model_saved", "image_model.pth"))
    verbose = config.get("verbose", 1)
    seed = config["feature_selection_options"].get("seed", 123)

    # Set Seed
    set_global_seed(seed)

    # Dataset & labels
    full_dataset = ImageDataset(image_dir, target_label, labels_csv, augment=False)  # Base dataset without augmentation
    labels_df = pd.read_csv(labels_csv)
    targets = labels_df[target_label].values  # Get 'label' column
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stratified K-Fold setup
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")

        # Create train and validation subsets
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        # Apply augmentation to the training subset if enabled
        if augmentation:
            train_subset.dataset = ImageDataset(image_dir, target_label, labels_csv, augment=True)

        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Model
        model = TransferLearningImageClassifier().to(device) if model_type == "DenseNet121" else CNN_Scratch().to(device)

        # Class Weights
        train_labels_list = [targets[i] for i in train_idx]
        if len(np.unique(train_labels_list)) == 2:
            class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=train_labels_list)
            pos_weight = torch.tensor([class_weights[1]], dtype=torch.float).to(device)
        else:
            pos_weight = torch.tensor([1.0], dtype=torch.float).to(device)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training for this fold
        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        all_labels, all_probs, all_preds = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        avg_val_loss = np.mean(val_losses)
        auc_score = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else None
        f1 = f1_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else None

        if verbose:
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")
            print(f"Val AUC: {auc_score if auc_score is not None else 'Not Computable'} | F1: {f1 if f1 is not None else 'Not Computable'}")

        fold_metrics.append({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": accuracy,
            "auc": auc_score,
            "f1": f1
        })

    # Aggregate Metrics
    print("\n--- K-Fold Results ---")
    for metric in ["train_loss", "val_loss", "accuracy", "auc", "f1"]:
        values = [m[metric] for m in fold_metrics if m[metric] is not None]
        if values:
            print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Retrain on Full Dataset
    print("\nRetraining on full dataset...")
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    final_model = TransferLearningImageClassifier().to(device) if model_type == "DenseNet121" else CNN_Scratch().to(device)
    if len(np.unique(targets)) == 2:
        class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=targets)
        pos_weight = torch.tensor([class_weights[1]], dtype=torch.float).to(device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        final_model.train()
        for images, labels in full_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = final_model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Full dataset training - Epoch {epoch+1}/{num_epochs} completed.")

    save_model(final_model, optimizer, model_save_path)
    print(f"Final model saved to {model_save_path}")

    return final_model