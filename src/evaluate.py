import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from dataset import get_training_data

DATA_PATH = os.path.join("data", "ucsd", "train")
MODEL_PATH = os.path.join("outputs", "models", "st_autoencoder.h5")
THRESHOLD_PATH = os.path.join("outputs", "models", "threshold.txt")
HISTOGRAM_PATH = os.path.join("outputs", "logs", "error_histogram.png")

def evaluate():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("Chưa có model!")
        return
    model = load_model(MODEL_PATH, compile=False)
    
    print("[INFO] Loading data để tính ngưỡng (64x64)...")
    # [CHANGED] Resize 64x64
    data = get_training_data(DATA_PATH, clip_len=10, resize=(64, 64)) 
    
    print("[INFO] Reconstructing clips...")
    reconstructed = model.predict(data, batch_size=4)
    
    # Tính MSE trên toàn bộ clip
    mse = np.mean(np.square(data - reconstructed), axis=(1, 2, 3, 4))
    
    plt.figure(figsize=(10, 6))
    plt.hist(mse, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title("Error Distribution (Spatiotemporal)")
    
    threshold = np.mean(mse) + 3 * np.std(mse)
    
    print(f"\n[RESULT] Mean MSE: {np.mean(mse)}")
    print(f"[RESULT] Threshold: {threshold}")
    
    plt.axvline(threshold, color='r', linestyle='dashed', label=f'Threshold: {threshold:.5f}')
    plt.savefig(HISTOGRAM_PATH)
    
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))
        
    print(f"[INFO] Đã lưu ngưỡng tại: {THRESHOLD_PATH}")

if __name__ == "__main__":
    evaluate()