import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import matplotlib.pyplot as plt


def preprocess_video(video_path, max_frames=60):
    # Frame extraction with timing
    start_time = time.time()

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
            frame = cv2.resize(frame, (112, 112))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame.astype(np.float32))
            frames.append(frame)
        count += 1

    cap.release()

    # Pad if needed
    while len(frames) < max_frames:
        frames.append(np.zeros((112, 112, 3), dtype=np.float32))

    frame_time = time.time() - start_time
    return np.array(frames), frame_time


def main():
    # Load Model
    model = load_model("best_goal_model.h5")

    # Get timings
    video_path = "data/test/test15_0.mov"
    video_input, frame_time = preprocess_video(video_path)

    # Modified Model with Intermediate Outputs for Timing
    mobilenet_layer = model.layers[0].layer  # Assuming MobileNet is wrapped in TimeDistributed
    lstm_layer = model.layers[2]  # LSTM Layer
    dense_layer = model.layers[-1]  # Final Dense (Classification) Layer

    cnn_model = tf.keras.models.Sequential([
        model.layers[0],
        model.layers[1]
    ])

    lstm_model = tf.keras.models.Sequential([
        model.layers[2],
        model.layers[3],
        model.layers[4]
    ])

    classification_model = tf.keras.models.Sequential([
        model.layers[-2],
        model.layers[-1]
    ])

    # 1. Time CNN Feature Extraction
    cnn_start = time.time()
    cnn_output = cnn_model.predict(np.expand_dims(video_input, axis=0))
    cnn_time = time.time() - cnn_start

    # 2. Time LSTM Processing
    lstm_start = time.time()
    lstm_output = lstm_model.predict(cnn_output)
    lstm_time = time.time() - lstm_start

    # 3. Time Classification Layer
    class_start = time.time()
    class_output = classification_model.predict(lstm_output)
    class_time = time.time() - class_start

    # Breakdown Visualization
    components = ['Frame Extraction', 'CNN', 'LSTM', 'Classification']
    times = [frame_time, cnn_time, lstm_time, class_time]

    plt.figure(figsize=(12, 5))

    # Pie Chart
    plt.subplot(1, 2, 1)
    plt.pie(times, labels=components, autopct='%1.1f%%',
            colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'], startangle=90)
    plt.title('Time Distribution')

    # Bar Chart
    plt.subplot(1, 2, 2)
    plt.bar(components, times, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    plt.ylabel('Seconds')
    plt.title('Component Timing Breakdown')

    plt.tight_layout()
    plt.savefig('inference_breakdown.png')
    plt.show()

    print(f"Goal Probability: {class_output[0][0]:.4f}")


if __name__ == "__main__":
    main()
