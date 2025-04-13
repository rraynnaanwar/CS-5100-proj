from tensorflow.keras.models import load_model
import time

model = load_model('best_goal_model.h5')

import cv2
import numpy as np


def preprocess_video(video_path, num_frames=16, img_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if i in frame_indices and ret:
            frame = cv2.resize(frame, img_size)
            frame = frame / 255.0  # normalize
            frames.append(frame)
    cap.release()

    if len(frames) < num_frames:
        # Pad with last frame if needed
        frames += [frames[-1]] * (num_frames - len(frames))

    return np.expand_dims(np.array(frames), axis=0)  # Shape: (1, N, 224, 224, 3)


video_input = preprocess_video("data/test/test16_1.mov")

# Predict probability
start = time.time()
prediction = model.predict(video_input)
print("Goal Probability:", prediction[0][0])
end = time.time()

print("time is ", (end - start) / 60, " mins")