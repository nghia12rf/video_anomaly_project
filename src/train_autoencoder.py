import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from dataset import get_training_data
from autoencoder import build_autoencoder

# --- CẤU HÌNH ---
# Đường dẫn dữ liệu (Bạn kiểm tra lại xem đúng folder chưa nhé)
DATA_PATH = os.path.join("data", "ucsd", "train") 
#DATA_PATH = os.path.join("data", "avenue", "train")
MODEL_SAVE_PATH = os.path.join("outputs", "models", "anomaly_detector.h5")
PLOT_SAVE_PATH = os.path.join("outputs", "logs", "training_plot.png")

# Tham số huấn luyện
# Để test nhanh thì để EPOCHS nhỏ (ví dụ 2). Khi train thật thì tăng lên 20-50.
EPOCHS = 10 
BATCH_SIZE = 32

def train():
    # 1. Tạo thư mục output nếu chưa có
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PLOT_SAVE_PATH), exist_ok=True)

    # 2. Load dữ liệu
    print("[INFO] Đang tải dữ liệu training...")
    data = get_training_data(DATA_PATH)
    
    if data is None or len(data) == 0:
        print("[ERROR] Không tìm thấy dữ liệu. Hãy kiểm tra lại folder data!")
        return

    # Shuffle dữ liệu để train tốt hơn
    np.random.shuffle(data)

    # Chia tập train/validation (80% train, 20% validate)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"[INFO] Training samples: {len(train_data)}")
    print(f"[INFO] Validation samples: {len(val_data)}")

    # 3. Xây dựng model
    print("[INFO] Đang khởi tạo model...")
    model = build_autoencoder()

    # 4. Cấu hình Callbacks (Tự động lưu model tốt nhất)
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # 5. Bắt đầu Training
    # Lưu ý: input là train_data, target cũng là train_data (vì là Autoencoder)
    print("[INFO] Bắt đầu train (Đi pha cà phê đợi xíu)...")
    history = model.fit(
        train_data, train_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(val_data, val_data),
        callbacks=[checkpoint, early_stopping]
    )

    # 6. Vẽ biểu đồ Loss
    print("[INFO] Đang vẽ biểu đồ training loss...")
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progress')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()
    print(f"[SUCCESS] Đã lưu biểu đồ tại: {PLOT_SAVE_PATH}")
    print(f"[SUCCESS] Đã lưu model tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()