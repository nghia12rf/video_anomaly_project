import cv2
import os
import glob
import sys

# Kiểm tra xem có thư viện tqdm chưa, nếu chưa có thì dùng hàm giả lập để không lỗi
try:
    from tqdm import tqdm
except ImportError:
    print("[WARN] Chưa cài thư viện 'tqdm' (thanh tiến trình). Đang dùng chế độ cơ bản...")
    # Hàm giả lập tqdm nếu chưa cài
    def tqdm(iterable, desc=""):
        return iterable

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Lấy đường dẫn gốc của dự án (thư mục cha của thư mục tools)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn đến folder chứa các folder ảnh Test (Test001, Test002...)
# Đảm bảo bạn đã giải nén dataset vào đúng vị trí này
UCSD_TEST_DIR = os.path.join(BASE_DIR, "data", "ucsd", "test")

# Đường dẫn thư mục sẽ chứa video đầu ra
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "ucsd_videos")

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

def images_to_video(source_folder, output_path, fps=20):
    """
    Hàm chuyển đổi một thư mục chứa ảnh .tif thành 1 file video .mp4
    """
    # Lấy danh sách ảnh và sắp xếp theo tên (để frame đúng thứ tự 001, 002...)
    images = sorted(glob.glob(os.path.join(source_folder, "*.tif")))
    
    if not images:
        # Thử tìm đuôi .jpg hoặc .png phòng trường hợp dữ liệu khác
        images = sorted(glob.glob(os.path.join(source_folder, "*.jpg"))) + \
                 sorted(glob.glob(os.path.join(source_folder, "*.png")))
    
    if not images:
        print(f"⚠️ [SKIP] Bỏ qua {os.path.basename(source_folder)}: Không tìm thấy ảnh.")
        return

    # Đọc ảnh đầu tiên để lấy kích thước chuẩn cho VideoWriter
    frame = cv2.imread(images[0])
    if frame is None:
        print(f"❌ [LỖI] Không đọc được ảnh: {images[0]}")
        return
        
    height, width, layers = frame.shape

    # Khởi tạo VideoWriter
    # Sử dụng codec 'mp4v' để tạo file mp4 tương thích tốt
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    folder_name = os.path.basename(source_folder)
    print(f"🎥 Đang xử lý: {folder_name} -> {os.path.basename(output_path)} ({len(images)} frames)")
    
    # Ghi từng ảnh vào video
    # Dùng tqdm để hiện thanh % chạy cho chuyên nghiệp
    for image_path in tqdm(images, desc=f"Converting {folder_name}"):
        img = cv2.imread(image_path)
        if img is not None:
            video.write(img)
    
    video.release()
    print("✅ Xong.\n")

def main():
    print(f"--- TOOL CHUYỂN ĐỔI UCSD SANG MP4 ---")
    print(f"📂 Nguồn dữ liệu: {UCSD_TEST_DIR}")
    print(f"📂 Nơi lưu video: {OUTPUT_DIR}")
    print("-" * 40)

    # Kiểm tra đường dẫn nguồn
    if not os.path.exists(UCSD_TEST_DIR):
        print(f"❌ [LỖI] Không tìm thấy đường dẫn: {UCSD_TEST_DIR}")
        print("👉 Vui lòng kiểm tra lại cấu trúc thư mục dự án.")
        return

    # Lấy danh sách tất cả các thư mục con trong ucsd/test
    all_items = os.scandir(UCSD_TEST_DIR)
    test_folders = []
    
    for item in all_items:
        if item.is_dir():
            # Chỉ lấy các folder tên là Test... (Bỏ qua folder _gt chứa ảnh mask)
            if "Test" in item.name and "_gt" not in item.name:
                test_folders.append(item.path)
    
    # Sắp xếp lại cho đẹp (Test001, Test002...)
    test_folders.sort()

    if not test_folders:
        print("⚠️ Không tìm thấy thư mục Test nào (Test001, Test002...).")
        return

    print(f"🔍 Tìm thấy {len(test_folders)} thư mục video cần chuyển đổi.")
    print("🚀 Bắt đầu chuyển đổi...\n")

    for folder in test_folders:
        folder_name = os.path.basename(folder)
        output_file = os.path.join(OUTPUT_DIR, f"{folder_name}.mp4")
        
        # Gọi hàm chuyển đổi
        images_to_video(folder, output_file)

    print("=" * 40)
    print(f"🎉 HOÀN TẤT! Video đã được lưu tại: {OUTPUT_DIR}")
    print("💡 Gợi ý: Hãy mở Gradio và upload các file video này để demo.")

if __name__ == "__main__":
    main()