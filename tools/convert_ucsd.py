import cv2
import os
import glob
import numpy as np
# Import MoviePy để tạo video chuẩn Web (H.264)
from moviepy.editor import ImageSequenceClip

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UCSD_TEST_DIR = os.path.join(BASE_DIR, "data", "ucsd", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ucsd_videos")

# Tạo thư mục đầu ra
os.makedirs(OUTPUT_DIR, exist_ok=True)

def images_to_video(source_folder, output_path, fps=20):
    # 1. Lấy danh sách ảnh
    images = sorted(glob.glob(os.path.join(source_folder, "*.tif")))
    
    # Fallback cho jpg/png
    if not images:
        images = sorted(glob.glob(os.path.join(source_folder, "*.jpg"))) + \
                 sorted(glob.glob(os.path.join(source_folder, "*.png")))

    if not images:
        print(f"[SKIP] Không có ảnh trong: {os.path.basename(source_folder)}")
        return

    print(f"🎥 Đang xử lý: {os.path.basename(source_folder)} -> MP4 (H.264)...")
    
    # 2. Đọc ảnh bằng OpenCV để đảm bảo không lỗi
    frames = []
    for img_path in images:
        # Đọc ảnh (OpenCV đọc rất mạnh file .tif)
        img = cv2.imread(img_path)
        
        if img is not None:
            # OpenCV đọc là BGR, MoviePy cần RGB. Phải convert!
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)
        else:
            print(f"⚠️ Cảnh báo: Không đọc được file {img_path}")

    if not frames:
        print("❌ Lỗi: Không đọc được frame nào.")
        return

    try:
        # 3. Đưa list ảnh đã chuẩn hóa vào MoviePy
        # Lúc này MoviePy nhận Numpy Array chuẩn RGB nên sẽ không bị lỗi Index nữa
        clip = ImageSequenceClip(frames, fps=fps)
        
        # Xuất file video chuẩn Web
        clip.write_videofile(
            output_path, 
            codec='libx264', 
            audio=False, 
            logger=None # Tắt log rác
        )
        
        print(f"✅ Xong: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"❌ Lỗi ghi video: {e}")

def main():
    if not os.path.exists(UCSD_TEST_DIR):
        print(f"❌ Không tìm thấy thư mục: {UCSD_TEST_DIR}")
        return

    subfolders = sorted([f.path for f in os.scandir(UCSD_TEST_DIR) if f.is_dir()])
    print(f"Tìm thấy {len(subfolders)} thư mục. Bắt đầu chuyển đổi...")

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        # Chỉ xử lý thư mục Test, bỏ qua _gt
        if "Test" in folder_name and "_gt" not in folder_name:
            output_file = os.path.join(OUTPUT_DIR, f"{folder_name}.mp4")
            images_to_video(folder, output_file)

    print(f"\n🎉 HOÀN TẤT! Video đã lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()