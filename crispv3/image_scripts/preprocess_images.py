from PIL import Image
import cv2
import numpy as np
import os
import argparse
import json
import albumentations as A
import pandas as pd
Image.MAX_IMAGE_PIXELS = None

### Sample Command to pre-process image dataset ###
# python image_scripts/preprocess_images.py --image_preprocess_config experiment_configs/image_preprocess.json

class Transformations:
    """Class containing modular image transformations and preprocessing pipeline."""
    
    def __init__(self, preprocess_config, image_size=(224, 224)):
        self.image_size = image_size
        self.transforms = {
            "horizontal_flip_transform": A.HorizontalFlip(p=1.0),
            "vertical_flip_transform": A.VerticalFlip(p=1.0),
            "rotate_90_transform": A.Rotate(limit=(90, 90), p=1.0),
            "brightness_contrast_transform": A.RandomBrightnessContrast(p=1.0),
            "gaussian_blur_transform": A.GaussianBlur(blur_limit=(3, 7), p=1.0), 
        }

        self.preprocess_config = preprocess_config

    def apply_transform(self, image, transform_name):
        """Apply a specific transformation to an image."""
        transform = self.transforms.get(transform_name)
        if transform:
            augmented = transform(image=image)
            return augmented["image"]
        else:
            raise ValueError(f"Transformation '{transform_name}' not found!")
    
    def transform_images(self):
        """Preprocess images by applying transformations and saving outputs."""
        image_folder = self.preprocess_config.get("image_folder", "img_input")
        output_root_folder = self.preprocess_config.get("preprocessed_output_folder", f"{image_folder}_processed")
        environments = self.preprocess_config.get("environments", "env_split")

        # If input directory exists then proceed with transformation steps!
        if os.path.isdir(image_folder):
            original_resized_folder = os.path.join(output_root_folder, "original_resized")
            os.makedirs(original_resized_folder, exist_ok=True)

            saved_images_df = pd.DataFrame(columns=['image_name', environments])
            saved_image_names, saved_env_split = list(), list()

            all_files = os.listdir(image_folder)
            # Check if folder empty or not
            if(len(all_files)>0):
                for filename in all_files:
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        print("Applying Transformations to images . . .")
                        file_path = os.path.join(image_folder, filename)
                        
                        try:
                            with Image.open(file_path) as img:
                                img_gray = img.convert("L")
                                img_array_gray = np.array(img_gray)

                                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # ADAPTIVE HE
                                equalized_img_array = clahe.apply(img_array_gray)
                                equalized_img = Image.fromarray(equalized_img_array)

                                intermediate_size = (equalized_img.size[0] // 4, equalized_img.size[1] // 4)
                                img_intermediate = equalized_img.resize(intermediate_size, Image.Resampling.NEAREST)
                                img_resized = img_intermediate.resize(self.image_size, Image.Resampling.NEAREST)
                                img_array = np.array(img_resized) / 255.0

                                output_image_path = os.path.join(original_resized_folder, f"{os.path.splitext(filename)[0]}.npy")
                                print('Saving', output_image_path)
                                np.save(output_image_path, img_array)
                                saved_image_names.append(os.path.join("original_resized", f"{os.path.splitext(filename)[0]}.npy"))
                                saved_env_split.append("original_resized")
                                
                                # Apply additional transformations on original pre-processed images
                                for transform_name in self.transforms.keys():
                                    transformed_img = self.apply_transform(np.array(img_resized), transform_name)
                                    transformed_img_resized = cv2.resize(transformed_img, self.image_size)
                                    transformed_img_array = transformed_img_resized / 255.0
                                    
                                    transform_folder = os.path.join(output_root_folder, transform_name)
                                    os.makedirs(transform_folder, exist_ok=True)
                                    
                                    output_image_path = os.path.join(transform_folder, f"{os.path.splitext(filename)[0]}.npy")
                                    np.save(output_image_path, transformed_img_array)
                                    print(f"Saved: {output_image_path}")

                                    saved_image_names.append(os.path.join(transform_name, f"{os.path.splitext(filename)[0]}.npy"))
                                    saved_env_split.append(transform_name)
                        
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")

                saved_images_df['image_name']=saved_image_names
                saved_images_df[environments]=saved_env_split
                image_names_save_path = os.path.join(output_root_folder, 'labels.csv')
                saved_images_df.sort_values(by=[environments, 'image_name'], ascending=[True, True], inplace=True)
                print('Saving image names csv', image_names_save_path)
                saved_images_df.to_csv(image_names_save_path, index=False)
            else:
                print(f'Warning: Directory {image_folder} is empty! Please input a valid directory with image files (.jpg or .jpeg or .png).')
        else:
            print(f'Warning: Directory {image_folder} is invalid! Please input a valid directory.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRISP Image Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Path to preprocess config json file
    parser.add_argument('--image_preprocess_config', default='experiment_configs/image_preprocess.json')
    opt = parser.parse_args()

    with open(os.path.join(os.getcwd(), opt.image_preprocess_config)) as json_file:
        preprocess_config = json.load(json_file)

    transformations = Transformations(preprocess_config)
    transformations.transform_images()