import os
import json

def validate_json_config(config):
    errors = []
    warnings = []

    # Check Required Top-Level Keys
    required_top_keys = ['image_data', 'data_options']
    for key in required_top_keys:
        if key not in config:
            errors.append(f"Missing top-level key: '{key}'")

    # Validate 'image_data' section
    image_data = config.get('image_data', {})
    required_image_keys = ['image_dir', 'labels_csv', 'gradcam_features_save_path']
    for key in required_image_keys:
        if key not in image_data:
            errors.append(f"Missing 'image_data' key: '{key}'")
        else:
            if key == 'image_dir':
                if not os.path.isdir(image_data[key]):
                    errors.append(f"Directory not found for '{key}': {image_data[key]}")
            elif key == 'labels_csv':
                if not os.path.isfile(image_data[key]):
                    errors.append(f"File not found for '{key}': {image_data[key]}")
                elif not image_data[key].endswith('.csv'):
                    errors.append(f"'{key}' should have .csv extension: {image_data[key]}")
            elif key == 'gradcam_features_save_path':
                if not (image_data[key].endswith('.pkl') or image_data[key].endswith('.pickle')):
                    errors.append(f"'gradcam_features_save_path' should have .pkl or .pickle extension: {image_data[key]}")
                parent_dir = os.path.dirname(image_data[key])
                if not os.path.isdir(parent_dir):
                    errors.append(f"Parent directory for 'gradcam_features_save_path' does not exist: {parent_dir}")

    # Optional: 'model_save_path'
    model_save_path = image_data.get('model_save_path')
    if model_save_path:
        if not model_save_path.endswith('.pth'):
            errors.append(f"'model_save_path' should have .pth extension: {model_save_path}")
        else:
            parent_dir = os.path.dirname(model_save_path)
            if not os.path.isdir(parent_dir):
                warnings.append(f"Parent directory for 'model_save_path' does not exist: {parent_dir} (but will be created during training).")
    else:
        warnings.append("Warning: 'model_save_path' not provided. Default will be used: image_model_saved/image_model.pth.")

    # Validate 'model_type'
    valid_model_types = ['DenseNet121', 'CNN_Scratch']
    model_type = image_data.get('model_type')
    if model_type:
        if model_type not in valid_model_types:
            errors.append(f"'model_type' should be one of {valid_model_types}, found '{model_type}'")
    else:
        warnings.append(f"'model_type' not specified in 'image_data'. Default will be used: 'DenseNet121'.")

    # Validate 'image_model_training_type'
    valid_training_types = ['train_test_split', 'full_loocv', 'k_fold']
    training_type = image_data.get('image_model_training_type')
    if training_type:
        if training_type not in valid_training_types:
            errors.append(f"'image_model_training_type' should be one of {valid_training_types}, found '{training_type}'")
    else:
        warnings.append(f"'image_model_training_type' not specified in 'image_data'. Default will be used: 'train_test_split'.")

    # Validate 'augmentation'
    if 'augmentation' in image_data:
        if not isinstance(image_data['augmentation'], bool):
            errors.append(f"'augmentation' should be a boolean, found {type(image_data['augmentation']).__name__}")

    # Validate 'image_model_gradcam'
    gradcam_config = image_data.get('image_model_gradcam', {})
    if gradcam_config:
        if 'apply_gradcam' in gradcam_config and not isinstance(gradcam_config['apply_gradcam'], bool):
            errors.append(f"'apply_gradcam' in 'image_model_gradcam' should be a boolean, found {type(gradcam_config['apply_gradcam']).__name__}")
        if 'gradcam_output_save_path' in gradcam_config:
            if not os.path.isdir(gradcam_config['gradcam_output_save_path']):
                warnings.append(f"'gradcam_output_save_path' directory does not exist: {gradcam_config['gradcam_output_save_path']} (will be created during execution if flag true).")

    # Validate 'gradcam_features_explainer'
    explainer_config = image_data.get('gradcam_features_explainer', {})
    if explainer_config:
        if 'save_path' in explainer_config:
            if not os.path.isdir(explainer_config['save_path']):
                warnings.append(f"'save_path' in 'gradcam_features_explainer' does not exist: {explainer_config['save_path']} (will be created during execution if flag true).")
        for boolean_field in ['show_clusters', 'show_com']:
            if boolean_field in explainer_config and not isinstance(explainer_config[boolean_field], bool):
                errors.append(f"'{boolean_field}' in 'gradcam_features_explainer' should be a boolean, found {type(explainer_config[boolean_field]).__name__}")

    # Validate 'data_options' section
    data_options = config.get('data_options', {})
    required_data_keys = ['subject_keys', 'targets', 'environments']
    for key in required_data_keys:
        if key not in data_options:
            errors.append(f"Missing 'data_options' key: '{key}'")

    # Validate 'dataset_fp'
    if data_options:
        dataset_fp = data_options.get('dataset_fp')
        if dataset_fp:
            if not (dataset_fp.endswith('.pkl') or dataset_fp.endswith('.pickle')):
                errors.append(f"'dataset_fp' should have .pkl or .pickle extension: {dataset_fp}")
            elif not os.path.isfile(dataset_fp):
                warnings.append(f"'dataset_fp' file not found: {dataset_fp}. This may be expected if running before data generation.")
        else:
            errors.append("Missing 'dataset_fp' in 'data_options'")

    # Hyperparameters
    hyperparams = {
        'split_ratio': float,
        'batch_size': int,
        'learning_rate': float,
        'num_epochs': int
    }
    for param, expected_type in hyperparams.items():
        if param in image_data:
            if not isinstance(image_data[param], expected_type):
                errors.append(f"'{param}' should be of type {expected_type.__name__}, found {type(image_data[param]).__name__}")
            else:
                if param == 'split_ratio' and not (0 < image_data[param] <= 1):
                    errors.append(f"'split_ratio' should be between 0 and 1, found {image_data[param]}")
                elif param != 'split_ratio' and image_data[param] < 0:
                    errors.append(f"'{param}' should be non-negative, found {image_data[param]}")

    # Validate 'tabular_features_path' for multimodal
    experiment_type = config.get('experiment_type', 'tabular_only')
    if experiment_type == 'multimodal':
        tabular_path = image_data.get('tabular_features_path')
        if tabular_path:
            if not (tabular_path.endswith('.pkl') or tabular_path.endswith('.pickle')):
                errors.append(f"'tabular_features_path' should have .pkl or .pickle extension: {tabular_path}")
            elif not os.path.isfile(tabular_path):
                errors.append(f"'tabular_features_path' file not found: {tabular_path}")
        else:
            errors.append("'tabular_features_path' is required in 'image_data' for multimodal experiments.")

    # Final reporting
    if errors:
        print("Config Validation Failed:")
        for err in errors:
            print("  -", err)
        exit(1)
    else:
        print("Config Validation Passed Successfully!")
        if warnings:
            print("\nWarnings:")
            for warn in warnings:
                print("  -", warn)
