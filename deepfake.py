"""
Team Name: teamname
File: train_model.py

This script performs the following steps:
  1. Scans the dataset directory (which contains 'fake' and 'real' subdirectories) for images.
  2. Assigns labels (0 for fake, 1 for real) to the images.
  3. Splits the data into training, validation, and test sets (60/20/20 split) using train_test_split.
  4. Creates TensorFlow datasets for all splits.
  5. Builds and trains an improved CNN model (using EfficientNetB0 with data augmentation) to classify the images.
  6. Evaluates and prints the model accuracy on training, validation, and test data.
  7. Computes and prints the F1 score on the test set.
"""
import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def get_image_paths_and_labels(dataset_path):
    """
    Scans the dataset directory for images in the subfolders 'fake' and 'real'
    and returns lists of file paths and corresponding labels.
    
    Expected directory structure:
        dataset/
            fake/   --> Contains fake images.
            real/   --> Contains real images.
    
    Labels:
      - 0 for fake
      - 1 for real
    """
    categories = ["fake", "real"]
    file_paths = []
    labels = []
    
    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_path, category)
        if not os.path.isdir(category_path):
            print(f"Directory {category_path} does not exist. Skipping.")
            continue
        
        for file in os.listdir(category_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(category_path, file)
                file_paths.append(full_path)
                labels.append(label)
    
    return file_paths, labels

def load_image(image_path, target_size=(224, 224)):
    """
    Reads and preprocesses an image:
      - Reads the image from disk.
      - Decodes it (assuming it's a JPEG/PNG image).
      - Resizes it to the target size.
      - Normalizes pixel values to the range [0, 1].
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, target_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def create_dataset(image_paths, labels, batch_size=32, target_size=(224, 224)):
    """
    Creates a tf.data.Dataset from image file paths and corresponding labels.
    """
    image_paths_tensor = tf.constant(image_paths)
    labels_tensor = tf.constant(labels, dtype=tf.int32)
    
    def _load_image_and_label(path, label):
        image = load_image(path, target_size)
        return image, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths_tensor, labels_tensor))
    dataset = dataset.map(_load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Builds an improved model using transfer learning with EfficientNetB0.
    
    Steps:
      1. Loads EfficientNetB0 as the base model (excluding the top layers) with ImageNet weights.
      2. Unfreezes the base model so that it can be fine-tuned from the start.
      3. Adds data augmentation, global average pooling, an extra Dense layer,
         and a custom classification head.
    """
    # Load the base model (without the top classifier layers)
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Unfreeze the base model so all layers are trainable
    base_model.trainable = True

    # Define the model architecture with a small data augmentation block
    inputs = tf.keras.Input(shape=input_shape)
    # Data augmentation
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    # Preprocessing specific to EfficientNetB0
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    # Pass through the base model (set training=True to ensure any dropout/batchnorm in base_model works in training mode)
    x = base_model(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile the model with a moderate learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # -------------------------
    # 1. Set the Dataset Path
    # -------------------------
    dataset_path = "dataset"  # Ensure this directory contains 'fake' and 'real' subfolders.
    
    # -------------------------
    # 2. Get Image Paths and Labels
    # -------------------------
    file_paths, labels = get_image_paths_and_labels(dataset_path)
    print(f"Total images found: {len(file_paths)}")
    
    if len(file_paths) == 0:
        print("No images found. Exiting.")
        return
    
    # -------------------------
    # 3. Split Data into Training, Validation, and Test Sets
    # -------------------------
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.4, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print("Number of training images:", len(train_paths))
    print("Number of validation images:", len(val_paths))
    print("Number of test images:", len(test_paths))
    
    # -------------------------
    # 4. Create TensorFlow Datasets
    # -------------------------
    batch_size = 32
    target_size = (224, 224)  # Increased target size for transfer learning
    train_dataset = create_dataset(train_paths, train_labels, batch_size, target_size)
    val_dataset = create_dataset(val_paths, val_labels, batch_size, target_size)
    test_dataset = create_dataset(test_paths, test_labels, batch_size, target_size)
    
    # -------------------------
    # 5. Build and Compile the Improved Model
    # -------------------------
    model = build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=2)
    model.summary()
    
    # -------------------------
    # 6. Train the Model
    # -------------------------
    epochs = 25  # Increase the number of epochs to allow the model to converge
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    
    # -------------------------
    # 7. Evaluate Model Accuracy on Training, Validation, and Test Data
    # -------------------------
    train_loss, train_accuracy = model.evaluate(train_dataset)
    print("Training Accuracy: {:.2f}%".format(train_accuracy * 100))
    
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print("Validation Accuracy: {:.2f}%".format(val_accuracy * 100))
    
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
    
    # -------------------------
    # 8. Compute F1 Score on the Test Set
    # -------------------------
    y_true = []
    y_pred = []
    for images, labels in test_dataset:
        predictions = model.predict(images)
        predictions = np.argmax(predictions, axis=1)
        y_pred.extend(predictions)
        y_true.extend(labels.numpy())
    
    f1 = f1_score(y_true, y_pred, average='weighted')
    print("Test F1 Score: {:.4f}".format(f1))

if __name__ == "__main__":
    main()
