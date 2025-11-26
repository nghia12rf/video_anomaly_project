import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from dataset import get_training_data

# --- CẤU HÌNH ---
DATA_PATH = os.path.join("data", "ucsd", "train")
#DATA_PATH = os.path.join("data", "avenue", "train")
MODEL_PATH = os.path.join("outputs", "models", "anomaly_detector.h5")
THRESHOLD_PATH = os.path.join("outputs", "models", "threshold.txt")
HISTOGRAM_PATH = os.path.join("outputs", "logs", "error_histogram.png")

def evaluate():
    # 1. Load Model
    print(f"[INFO] Đang tải model từ {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Chưa có model! Hãy chạy train_autoencoder.py trước.")
        return
    
    model = load_model(MODEL_PATH, compile=False)
    
    # 2. Load Dữ liệu Train (Dữ liệu bình thường)
    # Ta dùng chính tập train để xem model tái tạo nó "tốt" đến mức nào
    data = get_training_data(DATA_PATH)
    
    # 3. Dự đoán (Tái tạo lại ảnh)
    print("[INFO] Đang thực hiện tái tạo ảnh để tính lỗi...")
    reconstructed = model.predict(data)
    
    # 4. Tính Mean Squared Error (MSE) cho từng tấm ảnh
    # Công thức: Trung bình cộng của bình phương hiệu (Gốc - Tái tạo)
    mse = np.mean(np.square(data - reconstructed), axis=(1, 2, 3))
    
    # 5. Vẽ biểu đồ phân bố lỗi (Histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(mse, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title("Phân bố lỗi tái tạo (Reconstruction Error Distribution)")
    plt.xlabel("Mean Squared Error (MSE)")
    plt.ylabel("Số lượng Frame")
    
    # 6. Tính Ngưỡng (Threshold)
    # Cách chọn ngưỡng phổ biến: Mean + 3 * Std (Độ lệch chuẩn)
    # Nghĩa là: Chỉ 0.3% dữ liệu chuẩn bị coi nhầm là bất thường (Lý thuyết thống kê)
    threshold = np.mean(mse) + 3 * np.std(mse)
    
    print(f"\n[KẾT QUẢ] Lỗi trung bình (Mean): {np.mean(mse)}")
    print(f"[KẾT QUẢ] Ngưỡng đề xuất (Threshold): {threshold}")
    
    # Vẽ đường ngưỡng lên biểu đồ
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.5f}')
    plt.legend()
    plt.savefig(HISTOGRAM_PATH)
    print(f"[INFO] Đã lưu biểu đồ Histogram tại: {HISTOGRAM_PATH}")
    
    # 7. Lưu ngưỡng vào file text
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))
    print(f"[INFO] Đã lưu giá trị ngưỡng vào: {THRESHOLD_PATH}")

if __name__ == "__main__":
    evaluate()