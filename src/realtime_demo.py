import cv2
import numpy as np
import os
import glob
import time
from tensorflow.keras.models import load_model

# --- CẤU HÌNH ---
MODEL_PATH = os.path.join("outputs", "models", "anomaly_detector.h5")
THRESHOLD_PATH = os.path.join("outputs", "models", "threshold.txt")

# Chọn video để test (Bạn có thể đổi đường dẫn này)
# Ví dụ UCSD (Test001 chứa người đi xe đạp - Bất thường)
TEST_DATA_PATH = os.path.join("data", "ucsd", "test", "Test001") 

# Nếu muốn test Avenue (Video file)
# TEST_DATA_PATH = os.path.join("data", "avenue", "test", "01.avi")

def play_sound_alert():
    # Hàm phát tiếng kêu 'Beep' của hệ thống (Windows)
    import winsound
    winsound.Beep(1000, 200) # Tần số 1000Hz, 200ms

def main():
    # 1. Load Ngưỡng
    if not os.path.exists(THRESHOLD_PATH):
        print("[ERROR] Chưa có file ngưỡng (threshold.txt). Chạy evaluate.py trước!")
        return
    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read())
    print(f"[INFO] Đã load ngưỡng: {threshold}")

    # 2. Load Model (Thêm compile=False để tránh lỗi version)
    print("[INFO] Đang tải model...")
    model = load_model(MODEL_PATH, compile=False)

    # 3. Chuẩn bị nguồn video (Hỗ trợ cả Folder ảnh UCSD và Video file)
    frames = []
    is_video_file = False
    
    if os.path.isdir(TEST_DATA_PATH):
        # Nếu là folder (UCSD)
        image_paths = sorted(glob.glob(os.path.join(TEST_DATA_PATH, "*.tif")) + 
                             glob.glob(os.path.join(TEST_DATA_PATH, "*.jpg")))
        print(f"[INFO] Đang chạy demo trên folder ảnh: {len(image_paths)} frames")
        # Đọc trước các đường dẫn
        frames = image_paths
    else:
        # Nếu là file video (Avenue)
        print(f"[INFO] Đang chạy demo trên video file.")
        is_video_file = True
        cap = cv2.VideoCapture(TEST_DATA_PATH)

    # 4. Vòng lặp xử lý Realtime
    idx = 0
    while True:
        # Đọc frame
        if is_video_file:
            ret, frame = cap.read()
            if not ret: break
        else:
            if idx >= len(frames): break
            frame = cv2.imread(frames[idx])
            idx += 1
            
        if frame is None: break

        # Resize để hiển thị cho đẹp (zoom lên x2)
        display_frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
        
        # --- XỬ LÝ MODEL ---
        # 1. Tiền xử lý (giống hệt lúc train)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (128, 128))
        input_data = gray_resized.astype("float32") / 255.0
        input_data = np.expand_dims(input_data, axis=0)     # (1, 128, 128)
        input_data = np.expand_dims(input_data, axis=-1)    # (1, 128, 128, 1)
        
        # 2. Predict & Tính lỗi
        reconstructed = model.predict(input_data, verbose=0)
        mse = np.mean(np.square(input_data - reconstructed))
        
        # 3. So sánh với Ngưỡng
        label = "BINH THUONG"
        color = (0, 255, 0) # Xanh lá
        
        if mse > threshold:
            label = "CANH BAO: BAT THUONG!"
            color = (0, 0, 255) # Đỏ
            # play_sound_alert() # Bỏ comment dòng này nếu muốn nghe tiếng kêu
            
        # 4. Vẽ lên màn hình
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(display_frame, f"{label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_frame, f"Error: {mse:.5f} | Threshold: {threshold:.5f}", (10, display_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 5. Hiển thị
        cv2.imshow("Video Giam Sat (Nhan Q de thoat)", display_frame)
        
        # Hiện thêm ảnh tái tạo (để so sánh)
        recon_img = (reconstructed[0, :, :, 0] * 255).astype("uint8")
        cv2.imshow("AI 'Tuong tuong'", cv2.resize(recon_img, (200, 200)))

        # Chờ 30ms (giả lập tốc độ video), nhấn Q để thoát
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    if is_video_file: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()