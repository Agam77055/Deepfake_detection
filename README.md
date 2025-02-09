# Fake vs. Real Image Classification

## Overview
This project aims to build a robust image classification system capable of distinguishing between "fake" and "real" images. The development process evolved from a simple CNN to a complex transfer-learning model using EfficientNetB0, combined with rigorous data cleaning and preprocessing. This repository includes scripts for data preprocessing, model training, evaluation, inference, and generating prediction outputs in JSON format.

## Repository Structure
- **`train_model.py`**:  
  Contains code for scanning the dataset, assigning labels, splitting the data (60/20/20), building and training an improved CNN model using EfficientNetB0 with data augmentation, evaluating the model, and saving the trained model (`my_trained_model.h5`).

- **`predict_and_save_json.py`**:  
  Loads the saved model and generates predictions for a set of test images. The predictions are saved in a JSON file with the predicted label ("fake" or "real") for each image.

- **`image_separate.py`**:  
  Preprocesses the dataset by reading prediction JSON files to identify mislabeled images. It then moves images into separate folders (`real_images` and `fake_images`) based on their corrected labels. This script ensures that the dataset is properly cleaned before training.

- **`README.md`**:  
  This file, which describes the project, its structure, and usage instructions.

## Data Preprocessing and Cleaning
### Background
Our raw dataset was organized into two subdirectories:
- `fake/` – Expected to contain fake images.
- `real/` – Expected to contain real images.

During initial analysis, we discovered misclassified images (e.g., real images in the fake folder and vice versa). To correct this, we implemented an automated cleaning process.

### Image Separation with `image_separate.py`
The `image_separate.py` script reads prediction results from JSON files and separates images into the correct folders. Below is the code snippet for the preprocessing script:

```python
import os
import json
import shutil

# Paths to JSON files
json_file_1 = "real_cifake_preds.json"
json_file_2 = "fake_cifake_preds.json"  # Update with actual paths

# Paths to image folders
image_folder_1 = "real_cifake_images"
image_folder_2 = "fake_cifake_images"

# Paths for output
output_real_folder = "real_images"
output_fake_folder = "fake_images"

# Create output directories if they don’t exist
os.makedirs(output_real_folder, exist_ok=True)
os.makedirs(output_fake_folder, exist_ok=True)

# Function to load JSON and return a dictionary {index: label}
def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return {entry["index"]: entry["prediction"] for entry in data}

# Load labels from both JSON files
labels_1 = load_json(json_file_1)
labels_2 = load_json(json_file_2)

# Process images from both folders
image_counter = {"real": 0, "fake": 0}  # To ensure unique naming

for folder, labels in [(image_folder_1, labels_1), (image_folder_2, labels_2)]:
    for index, label in labels.items():
        image_name = f"{index}.png"  # Construct the image filename
        src_path = os.path.join(folder, image_name)

        if label.lower() == "real":
            dest_folder = output_real_folder
            image_counter["real"] += 1
            new_filename = f"real_{image_counter['real']}.png"  # Avoid duplicate names
        elif label.lower() == "fake":
            dest_folder = output_fake_folder
            image_counter["fake"] += 1
            new_filename = f"fake_{image_counter['fake']}.png"
        else:
            continue  # Ignore unrecognized labels
        
        dest_path = os.path.join(dest_folder, new_filename)

        # Move the file if it exists
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)

print("All images have been successfully separated!")
