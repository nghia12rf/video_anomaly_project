import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dataset import get_training_data
from autoencoder import build_autoencoder

# --- CẤU HÌNH ---
DATA_PATH = os.path.join("data", "ucsd", "train") 
# DATA_PATH = os.path.join("data", "avenue", "train")

MODEL_SAVE_PATH = os.path.join("outputs", "models", "st_autoencoder.h5")
PLOT_SAVE_PATH = os.path.join("outputs", "logs", "training_plot.png")

EPOCHS = 20
BATCH_SIZE = 4 

def train():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)

    print("[INFO] Đang tải dữ liệu training (dạng clip 10 frames, size 64x64)...")
    # [CHANGED] Resize về 64x64 để tiết kiệm RAM
    data = get_training_data(DATA_PATH, clip_len=10, resize=(64, 64)) 
    
    if data is None: return

    np.random.shuffle(data)
    split_idx = int(len(data) * 0.9) # 90% train, 10% val
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"[INFO] Train shape: {train_data.shape}")
    print(f"[INFO] Val shape: {val_data.shape}")

    print("[INFO] Khởi tạo Spatiotemporal Model...")
    model = build_autoencoder(input_shape=(10, 64, 64, 1)) # [CHANGED] Update input shape

    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    print("[INFO] Bắt đầu train...")
    history = model.fit(
        train_data, train_data, 
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(val_data, val_data),
        callbacks=[checkpoint, early_stopping]
    )

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Spatiotemporal Model Loss')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()
    print(f"[SUCCESS] Đã lưu model tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()