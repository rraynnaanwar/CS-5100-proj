import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import datetime
import  time

# GPU Check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU is available and configured.")
else:
    print("GPU not found! Running on CPU.")

# Augmentation Function
def augment_frame(frame):
    if np.random.rand() < 0.5:
        frame = cv2.flip(frame, 1)
    if np.random.rand() < 0.5:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * (0.5 + np.random.rand())
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-10, 10)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        frame = cv2.warpAffine(frame, M, (w, h))
    return frame

# Frame Extraction Function
def extract_frames(video_path, max_frames=60, size=(112, 112), augment=False):
    print(f"Extracting frames from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    count = 0
    while len(frames) < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if augment:
                frame = augment_frame(frame)
            frame = preprocess_input(frame.astype(np.float32))
            frames.append(frame)
        count += 1
    cap.release()
    while len(frames) < max_frames:
        frames.append(np.zeros((size[1], size[0], 3), dtype=np.float32))
    print(f"Extracted {len(frames)} frames.")
    return np.array(frames)

# Load Data From CSV
def load_data_from_csv(csv_path, augment=False):
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total records: {len(df)}")
    X, y = [], []
    for idx, row in df.iterrows():
        print(f"Loading {idx + 1}/{len(df)}: {row['video_path']}")
        frames = extract_frames(row['video_path'], augment=augment)
        X.append(frames)
        y.append(row['label'])
    return np.array(X), np.array(y)

# Build Model with Transfer Learning
def build_model(input_shape=(30, 112, 112, 3)):
    print("Building model with MobileNetV2...")
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(112, 112, 3))
    base_cnn.trainable = False

    model = models.Sequential([
        layers.TimeDistributed(base_cnn, input_shape=input_shape),
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),
        layers.LSTM(64, return_sequences=False, dropout=0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print("Model ready!")
    return model

# Main Execution
if __name__ == "__main__":
    start = time.time()
    csv_path = "updated_file.csv"

    print("Loading and preprocessing dataset...")
    X, y = load_data_from_csv(csv_path, augment=True)
    print(f"Loaded data: {X.shape}, Labels: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split done — Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print("Building model...")
    model = build_model(input_shape=X_train.shape[1:])

    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_cb = ModelCheckpoint("best_goal_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")

    print("Training model (initial frozen CNN)...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=2,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_cb, checkpoint_cb]
    )

    print("Loading best model for evaluation...")
    best_model = tf.keras.models.load_model("best_goal_model.h5")
    loss, acc = best_model.evaluate(X_test, y_test)
    print(f"Best Model Accuracy: {acc:.2f}, Loss: {loss:.4f}")

    # Save the model as the final version
    best_model.save("goal_prediction_finetuned_model.h5")
    print("Model saved as goal_prediction_finetuned_model.h5")

    # === FINE-TUNING SECTION
    """
    print("Unfreezing top 20 layers of MobileNetV2 for fine-tuning...")
    base_cnn = best_model.layers[0].layer
    base_cnn.trainable = True
    for layer in base_cnn.layers[:-20]:
        layer.trainable = False

    best_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    print("Fine-tuning the model...")
    fine_tune_history = best_model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=2,
        validation_data=(X_test, y_test),
        callbacks=[tensorboard_cb]
    )

    print("Evaluating fine-tuned model...")
    loss, acc = best_model.evaluate(X_test, y_test)
    print(f"Final Accuracy After Fine-Tuning: {acc:.2f}, Loss: {loss:.4f}")

    best_model.save("goal_prediction_finetuned_model.h5")
    print("Final model saved as goal_prediction_finetuned_model.h5")
    """
    end = time.time()

    print("time is ", (end - start)/60," mins")