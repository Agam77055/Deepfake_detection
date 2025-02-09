# Fake vs. Real Image Classification

## Overview
This project is focused on building a robust image classification system to distinguish between "fake" and "real" images. The project evolved from an initial simple CNN to a more complex model using transfer learning with EfficientNetB0, accompanied by extensive data cleaning and preprocessing.

## Repository Structure
- **train_model.py**  
  - **Purpose:**  
    Scans the dataset, assigns labels, splits the data (60/20/20), creates TensorFlow datasets, builds and trains an improved CNN model (using EfficientNetB0 with data augmentation), evaluates the model, computes the F1 score, and saves the final trained model.
  - **Usage:**  
    Run this script to train the model on your cleaned dataset. The trained model is saved as `my_trained_model.h5`, which you can later load for making predictions.

- **predict_and_save_json.py**  
  - **Purpose:**  
    Loads the saved model and applies it to a set of test images (named sequentially such as `1.png`, `2.png`, etc.) located in a designated test folder. It then generates a JSON file (`predictions.json`) containing the predicted labels ("fake" or "real") for each image.
  - **Usage:**  
    After training and saving the model, run this script to produce predictions for new images.

- **image_separate.py**  
  - **Purpose:**  
    Processes the dataset by reading in prediction results from JSON files. It separates images into the correct folders based on their labels—moving images to either `real_images` or `fake_images` folders. This step ensures that the dataset is accurately cleaned (e.g., moving misclassified images to the appropriate folder) before model training.
  - **Usage:**  
    Run this script to clean and reorganize your images based on pre-existing prediction outputs, ensuring that the dataset used for training is of high quality.

- **README.md**  
  - **Purpose:**  
    Provides an overview of the project, including the purpose, structure, usage instructions, and links to the full project repository on GitHub.

## Data Preprocessing and Cleaning
Initially, the raw dataset was organized into two subdirectories (`fake/` and `real/`), but we discovered inconsistencies (e.g., real images in the fake folder and vice versa). We developed an automated cleaning process:
- **Automated Separation:**  
  A script (`image_separate.py`) reads prediction results from JSON files generated by other experiments. It then moves images to the correct folders (creating `real_images` and `fake_images`) based on these predictions.
- **Label Assignment:**  
  After cleaning, labels are assigned programmatically: `0` for fake images and `1` for real images.
- **Preprocessing:**  
  Each image is read, decoded, resized to 224×224 pixels, and preprocessed using EfficientNet’s preprocessing function before being fed into the model.

## Model Training
The `train_model.py` script carries out the following:
- **Data Loading and Splitting:**  
  The cleaned dataset is scanned, and images are split into training, validation, and test sets.
- **Building the Model:**  
  A transfer learning model is constructed using EfficientNetB0 as the base. Data augmentation layers (e.g., random flipping and rotation) are applied before the base model, and a classification head is added afterward.
- **Training and Evaluation:**  
  The model is trained for 25 epochs using the Adam optimizer. It is then evaluated on the training, validation, and test sets, and performance metrics (including the F1 score) are computed.
- **Model Saving:**  
  The trained model is saved as `my_trained_model.h5` for future inference.

## Inference and Prediction
The `predict_and_save_json.py` script is used after training to:
- Load the saved model.
- Preprocess new test images using the same methods as in training.
- Generate predictions, mapping the output (0 or 1) to "fake" or "real".
- Save the predictions in a JSON file (`predictions.json`), where each entry includes the image index and its predicted label.

## How to Run the Project
1. **Data Cleaning:**  
   Run `image_separate.py` to process your dataset and separate images into the correct folders.
2. **Model Training:**  
   Run `train_model.py` to train the model on the cleaned dataset and save the trained model.
3. **Inference:**  
   Run `predict_and_save_json.py` to generate predictions for new test images and create a JSON output.

## Dependencies
- Python 3.x
- Numpy
- TensorFlow 2.x
- scikit-learn  
