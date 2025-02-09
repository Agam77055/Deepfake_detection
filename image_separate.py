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
