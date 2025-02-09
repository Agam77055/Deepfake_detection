# Fake vs. Real Image Classification

## Overview
This project implements a robust image classification system to distinguish between "fake" and "real" images using transfer learning with EfficientNetB0. The entire workflow—from data preprocessing and cleaning to model training and inference—is implemented within a Jupyter Notebook. Additionally, a separate Python script (`image_separate.py`) is provided to clean the dataset by separating images based on their predicted labels.

## Repository Structure
- **`project.ipynb`**  
  The main Jupyter Notebook containing all code for:
  - Data loading and preprocessing.
  - Model building and training using EfficientNetB0 with data augmentation.
  - Model evaluation (accuracy and F1 score calculation).
  - Inference and JSON extraction, which generates a JSON file listing predictions (each entry contains an image index and its predicted label).
  
- **`image_separate.py`**  
  A Python script that reads prediction JSON files and separates images into two folders (`real_images` and `fake_images`) based on their labels. This preprocessing step ensures that the dataset is correctly cleaned before model training.

- **`README.md`**  
  This file, which provides an overview of the project, instructions for running the code, and details about the repository.

## Data Preprocessing and Cleaning
Initially, the raw dataset was divided into two folders (`fake/` and `real/`), but inconsistencies were discovered (e.g., real images in the fake folder and vice versa). To address this:
- A dedicated script (`image_separate.py`) was developed to read prediction results from JSON files.
- Based on these predictions, images are moved to the correct directories (`real_images` for real images and `fake_images` for fake images).
- This ensures that the dataset used for training is of high quality.

## Model Training and Inference
The `project.ipynb` notebook contains the complete pipeline:
1. **Data Loading & Splitting:**  
   - The notebook scans the cleaned dataset, assigns labels (0 for fake, 1 for real), and splits the data into training, validation, and test sets (60/20/20 split).
   
2. **Model Building:**  
   - A transfer learning model is constructed using EfficientNetB0 (pre-trained on ImageNet) as the base.
   - Data augmentation (e.g., random flips and rotations) is applied to improve model generalization.
   - A classification head (with dense and dropout layers) is appended to the base model.

3. **Training & Evaluation:**  
   - The model is trained for 25 epochs using the Adam optimizer.
   - The notebook evaluates model performance on training, validation, and test sets and calculates the F1 score.
   - The final model is saved (e.g., as `my_trained_model.h5`) for later inference.

4. **Inference & JSON Extraction:**  
   - The saved model is loaded to predict labels for new test images (named sequentially as `1.png`, `2.png`, …).
   - Predictions are mapped (0 → "fake", 1 → "real") and saved in a JSON file (`predictions.json`).

## How to Run the Project

1. **Data Cleaning (Optional):**
   - If your dataset requires cleaning, run the `image_separate.py` script:
     ```bash
     python image_separate.py
     ```
   - This script will organize images into `real_images` and `fake_images` folders based on prediction outputs.

2. **Model Training and Inference:**
   - Open the `project.ipynb` Jupyter Notebook.
   - Follow the instructions in the notebook, executing the cells sequentially.
   - The notebook covers the entire process from data preprocessing and model training to generating a JSON file of predictions.

## Dependencies
- Python 3.x
- Jupyter Notebook or JupyterLab
- TensorFlow 2.x
- scikit-learn
- Other built-in libraries: `os`, `json`, `shutil`

You can install the necessary Python packages using:
```bash
pip install tensorflow scikit-learn
