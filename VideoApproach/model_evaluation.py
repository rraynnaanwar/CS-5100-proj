import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras.models import load_model
from tqdm import tqdm

# === CONFIG ===
CSV_PATH = "test_files.csv"
VIDEO_ROOT = ""
MODEL_PATH = "best_goal_model.h5"
NUM_FRAMES = 16
IMG_SIZE = (224, 224)

# === Load model ===
model = load_model(MODEL_PATH)

# === Preprocessing function ===
def preprocess_video(video_path, num_frames=16, img_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []

    for i in range(total_frames):
        ret, frame = cap.read()
        if i in frame_indices and ret:
            frame = cv2.resize(frame, img_size)
            frame = frame / 255.0
            frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        if frames:
            frames += [frames[-1]] * (num_frames - len(frames))
        else:
            raise ValueError(f"No frames read from video: {video_path}")

    return np.expand_dims(np.array(frames), axis=0)

# === Load test CSV ===
df = pd.read_csv(CSV_PATH)

y_true = []
y_pred = []
y_probs = []

print("\n🔍 Starting Evaluation...\n")

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        abs_video_path = os.path.join(VIDEO_ROOT, row["video_path"])
        video_input = preprocess_video(abs_video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE)

        prob = model.predict(video_input)[0][0]
        pred = int(prob >= 0.5)

        y_probs.append(prob)
        y_pred.append(pred)
        y_true.append(int(row["label"]))

    except Exception as e:
        print(f"❌ Error processing {row['video_path']}: {e}")

# === Evaluation Metrics ===
if len(y_true) == 0:
    print("\n❌ No valid predictions were made. Please check the video paths in your CSV.")
else:
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\n📄 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Goal", "Goal"]))

    try:
        roc_auc = roc_auc_score(y_true, y_probs)
        print(f"\n🔥 ROC AUC Score: {roc_auc:.4f}")

        # === Plot ROC Curve ===
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Goal Prediction")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not compute ROC AUC: {e}")
