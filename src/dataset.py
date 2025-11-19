import cv2
import numpy as np
import glob
import os
import sys

def preprocess_frame(frame, resize=(128, 128)):
    """
    Hàm chuẩn hóa khung hình:
    1. Chuyển sang ảnh xám (Grayscale).
    2. Resize về kích thước cố định (128x128).
    3. Chuẩn hóa pixel về đoạn [0, 1].
    """
    # Chuyển sang ảnh xám
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
        
    # Resize
    gray_resized = cv2.resize(gray, resize)
    
    # Chuẩn hóa (Normalize)
    normalized = gray_resized.astype("float32") / 255.0
    return normalized

def load_image_sequence(folder_path, resize=(128, 128)):
    """Đọc chuỗi ảnh từ folder (Dành cho UCSD)"""
    # Tìm tất cả file ảnh (tif, jpg, png)
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.tif")) + 
                         glob.glob(os.path.join(folder_path, "*.jpg")) +
                         glob.glob(os.path.join(folder_path, "*.png")))
    
    frames = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        processed = preprocess_frame(img, resize)
        frames.append(processed)
        
    return np.array(frames)

def load_video_file(video_path, resize=(128, 128)):
    """Đọc file video đơn lẻ (Dành cho Avenue)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = preprocess_frame(frame, resize)
        frames.append(processed)
    cap.release()
    return np.array(frames)

def get_training_data(root_dir, resize=(128, 128)):
    """
    Hàm chính để load dữ liệu. Tự động phát hiện loại dữ liệu.
    """
    print(f"[DATA] Đang quét dữ liệu tại: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"[ERROR] Không tìm thấy đường dẫn: {root_dir}")
        return None

    all_frames = []
    
    # 1. Kiểm tra xem có phải Avenue (chứa file video) không?
    videos = sorted(glob.glob(os.path.join(root_dir, "*.avi")) + 
                    glob.glob(os.path.join(root_dir, "*.mp4")))
    
    # 2. Kiểm tra xem có phải UCSD (chứa thư mục con) không?
    subfolders = sorted([f.path for f in os.scandir(root_dir) if f.is_dir()])

    if len(videos) > 0:
        print(f"[DATA] Phát hiện {len(videos)} video files (Avenue mode).")
        for v_path in videos:
            print(f"  -> Loading: {os.path.basename(v_path)}")
            frames = load_video_file(v_path, resize)
            all_frames.extend(frames)

    elif len(subfolders) > 0:
        print(f"[DATA] Phát hiện {len(subfolders)} thư mục chuỗi ảnh (UCSD mode).")
        for folder in subfolders:
            # Bỏ qua các folder _gt nếu lỡ còn sót lại
            if "_gt" in folder: 
                continue
            print(f"  -> Loading: {os.path.basename(folder)}")
            frames = load_image_sequence(folder, resize)
            all_frames.extend(frames)
            
    else:
        print("[WARNING] Thư mục rỗng hoặc cấu trúc không đúng!")
        return None

    # Chuyển list thành numpy array
    all_frames = np.array(all_frames)
    
    # Thêm chiều channel (N, 128, 128, 1) để khớp input của Keras
    if len(all_frames.shape) == 3:
        all_frames = np.expand_dims(all_frames, axis=-1)

    print(f"[DATA] Load hoàn tất. Shape dữ liệu: {all_frames.shape}")
    return all_frames

# --- Test Block ---
if __name__ == "__main__":
    # Test thử với UCSD
    print("--- TESTING UCSD ---")
    ucsd_path = os.path.join("data", "ucsd", "train")
    if os.path.exists(ucsd_path):
        get_training_data(ucsd_path)
    else:
        print("Chưa có dữ liệu UCSD để test")

    # Test thử với Avenue (nếu có)
    print("\n--- TESTING AVENUE ---")
    avenue_path = os.path.join("data", "avenue", "train")
    if os.path.exists(avenue_path):
        get_training_data(avenue_path)