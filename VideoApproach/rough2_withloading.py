import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import time

# Configuration
FRAME_COUNT = 75  # Adjust the frame count, it will be bad if this isnt accurate
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 25  # Start with more epochs
CSV_PATH = 'updated_file.csv'
TEST_VIDEO_PATH = 'data/goal/goal1.mov'  # Change to a valid path for testing

print(f"CSV_PATH: {CSV_PATH}")
print(f"TEST_VIDEO_PATH: {TEST_VIDEO_PATH}")


# Video Preprocessor
class GoalVideoProcessor:
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels
        print(f"GoalVideoProcessor initialized with {len(video_paths)} video paths and {len(labels)} labels")

    def process_video(self, path):
        print(f"Processing video: {path}")
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Error: Could not open video {path}")
                return None

            frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video or error reading frame from {path}")
                    break

                # Preprocessing
                frame = cv2.resize(frame, IMG_SIZE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                if len(frames) == FRAME_COUNT:
                    print(f"Reached frame count of {FRAME_COUNT} for {path}")
                    break

            cap.release()
            if len(frames) == FRAME_COUNT:
                print(f"Successfully processed {FRAME_COUNT} frames from {path}")
                return np.array(frames)
            else:
                print(f"Warning: Could not read enough frames from {path}")
                return None
        except Exception as e:
            print(f"Error processing video {path}: {e}")
            return None

    def data_generator(self):
        print("Data generator started")
        while True:
            paths = []
            labels = []
            for i in range(0, len(self.video_paths), BATCH_SIZE):
                batch_paths = self.video_paths[i:i + BATCH_SIZE]
                batch_labels = self.labels[i:i + BATCH_SIZE]
                print(f"Processing batch {i // BATCH_SIZE + 1}/{len(self.video_paths) // BATCH_SIZE + 1}")

                for video_path, label in zip(batch_paths, batch_labels):
                    print(f"Processing video {video_path} with label {label}")
                    video = self.process_video(video_path)
                    if video is not None:  # Check for successful processing
                        paths.append(video)
                        labels.append(label)
                        print(f"Added video {video_path} to batch")

                if paths:
                    print(f"Yielding batch of size {len(paths)}")
                    yield np.array(paths), np.array(labels)
                else:
                    print("Warning: No valid videos in this batch")


# Updated Model Architecture
def build_goal_net():
    print("Building goal net model")
    base_cnn = tf.keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_cnn.trainable = False  # Transfer learning

    model = models.Sequential([
        layers.Input(shape=(FRAME_COUNT, IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.TimeDistributed(base_cnn),
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    print("Goal net model built successfully")
    return model


# Updated Data Augmentation
def build_augmentation():
    print("Building data augmentation model")
    model = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.1)
    ])
    print("Data augmentation model built successfully")
    return model


# Training Setup
def train_model(train_paths, train_labels, val_paths, val_labels):
    print("Starting training model")
    model = build_goal_net()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    print("Model compiled successfully")

    processor = GoalVideoProcessor(train_paths, train_labels)
    print("GoalVideoProcessor initialized for training")
    val_processor = GoalVideoProcessor(val_paths, val_labels)
    print("GoalVideoProcessor initialized for validation")
    print(f"Current Working Directory: {os.getcwd()}")

    model.fit(
        processor.data_generator(),
        steps_per_epoch=max(1, len(train_paths) // BATCH_SIZE),  # Ensure at least one step
        epochs=EPOCHS,
        validation_data=val_processor.data_generator(),  # ADDED LINE
        validation_steps=max(1, len(val_paths) // BATCH_SIZE),  # ADDED LINE
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),  # Can be adjusted
            tf.keras.callbacks.ModelCheckpoint('goal_model.h5', save_best_only=True)
        ]
    )
    print("Model training completed")
    return model


# Inference System
class GoalPredictor:
    def __init__(self, model_path):
        print(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.window_size = FRAME_COUNT
        print("GoalPredictor initialized")

    def predict_goal_probability(self, video_path):
        print(f"Predicting goal probability for {video_path}")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return 0.0
            probabilities = []

            while True:
                frames = []
                while len(frames) < self.window_size:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"End of video or error reading frame from {video_path}")
                        break
                    frame = cv2.resize(frame, IMG_SIZE)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

                if len(frames) < self.window_size:
                    break

                frames = np.array(frames) / 255.0  # Normalize
                pred = self.model.predict(np.expand_dims(frames, 0))
                probabilities.append(float(pred[0][0]))

            cap.release()
            if probabilities:
                max_prob = np.max(probabilities)
                print(f"Max goal probability: {max_prob:.2f}")
                return max_prob  # Return peak probability
            else:
                print(f"Warning: No probabilities calculated for {video_path}")
                return 0.0
        except Exception as e:
            print(f"Error predicting goal probability for {video_path}: {e}")
            return 0.0  # Handle prediction error


# Updated Load Data Function
def load_data(csv_path):
    print(f"Loading data from CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path)  # Let pandas infer the header, since it exists
        # Debugging output
        print("CSV Loaded Successfully!")
        print("Sample Data:")
        print(df.head())

        # Ensure the data is well-formed
        df = df.dropna()  # Remove rows with missing values

        # Separate file paths and labels
        video_paths = df['video_path'].tolist()
        labels = df['label'].tolist()

        print(f"Loaded {len(video_paths)} video paths and {len(labels)} labels")

        # Print the first video_path and test if it exists
        if video_paths:
            first_path = video_paths[0]
            print(f"First video path: {first_path}")
            print(f"Path exists: {os.path.exists(first_path)}")

            # Print the absolute path to help debug
            print(f"Absolute path: {os.path.abspath(first_path)}")

        # Split into training and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            video_paths, labels, test_size=0.2, random_state=42
        )

        print(f"Training set size: {len(train_paths)}")
        print(f"Validation set size: {len(val_paths)}")

        # Convert data to list (already done)
        return train_paths, val_paths, train_labels, val_labels

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return [], [], [], []


# Main execution
if __name__ == "__main__":
    # Load data from CSV
    start = time.time()
    train_paths, val_paths, train_labels, val_labels = load_data(CSV_PATH)

    if not train_paths or not val_paths:
        print("Error: No training or validation data loaded. Check the CSV file and video paths.")
    else:
        # Remove non-existent paths before training
        train_paths = [path for path in train_paths if os.path.exists(path)]
        val_paths = [path for path in val_paths if os.path.exists(path)]

        print(f"Number of valid training paths after checking existence: {len(train_paths)}")
        print(f"Number of valid validation paths after checking existence: {len(val_paths)}")
        if not train_paths:
            print(
                "Error: No valid training paths found. Check that the paths in your CSV are correct and that the files exist.")
        # Train model
        model = train_model(train_paths, train_labels, val_paths, val_labels)

        # Example prediction on a test video
        # predictor = GoalPredictor('goal_model.h5') # removed
        # probability = predictor.predict_goal_probability(TEST_VIDEO_PATH) # removed

        # print(f"Goal Probability: {probability:.2%}") # removed

        # print the model summary to inspect the model
        model.summary()
    end = time.time()

    print("time is ", (end - start)/60," mins")