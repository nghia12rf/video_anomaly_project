import cv2
import numpy as np
import glob
import os

def preprocess_frame(frame, resize=(64, 64)): # [CHANGED] 128 -> 64
    """
    Chuẩn hóa frame: Grayscale -> Resize -> Normalize
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    gray_resized = cv2.resize(gray, resize)
    normalized = gray_resized.astype("float32") / 255.0
    return normalized

def create_clips(frames, clip_len=10):
    """
    Tạo các đoạn clip ngắn từ chuỗi frame (Sliding Window).
    Input: (N, 64, 64, 1)
    Output: (N-clip_len, 10, 64, 64, 1)
    """
    if len(frames) < clip_len:
        return np.array([])
        
    # Bỏ chiều channel tạm thời để xử lý
    frames = np.squeeze(frames)
    
    clips = []
    # Trượt cửa sổ: Frame 0-9, Frame 1-10...
    for i in range(len(frames) - clip_len + 1):
        clip = frames[i : i+clip_len]
        clips.append(clip)
        
    clips = np.array(clips)
    # Thêm lại chiều channel: (Batch, 10, 64, 64, 1)
    clips = np.expand_dims(clips, axis=-1)
    return clips

def load_video_data(path, is_folder=True, resize=(64, 64)): # [CHANGED] 128 -> 64
    frames = []
    if is_folder:
        # Load UCSD (Folder ảnh)
        image_paths = sorted(glob.glob(os.path.join(path, "*.tif")) + 
                             glob.glob(os.path.join(path, "*.jpg")) +
                             glob.glob(os.path.join(path, "*.png")))
        for p in image_paths:
            img = cv2.imread(p)
            if img is not None:
                frames.append(preprocess_frame(img, resize))
    else:
        # Load Avenue (Video file)
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(preprocess_frame(frame, resize))
        cap.release()
    
    return np.array(frames)

def get_training_data(root_dir, clip_len=10, resize=(64, 64)): # [CHANGED] 128 -> 64
    print(f"[DATA] Đang quét dữ liệu tại: {root_dir}")
    if not os.path.exists(root_dir):
        print(f"[ERROR] Không tìm thấy: {root_dir}")
        return None

    all_clips = []
    
    # 1. Check Avenue (Video files)
    videos = sorted(glob.glob(os.path.join(root_dir, "*.avi")) + 
                    glob.glob(os.path.join(root_dir, "*.mp4")))
    
    # 2. Check UCSD (Subfolders)
    subfolders = sorted([f.path for f in os.scandir(root_dir) if f.is_dir()])

    if len(videos) > 0:
        for v in videos:
            print(f"  -> Loading video: {os.path.basename(v)}")
            frames = load_video_data(v, is_folder=False, resize=resize)
            video_clips = create_clips(frames, clip_len)
            if len(video_clips) > 0:
                all_clips.append(video_clips)

    elif len(subfolders) > 0:
        for folder in subfolders:
            if "_gt" in folder: continue
            print(f"  -> Loading folder: {os.path.basename(folder)}")
            frames = load_video_data(folder, is_folder=True, resize=resize)
            folder_clips = create_clips(frames, clip_len)
            if len(folder_clips) > 0:
                all_clips.append(folder_clips)
    
    if len(all_clips) == 0:
        print("[WARNING] Không tạo được clip nào!")
        return None

    # Gộp tất cả clip lại
    final_data = np.concatenate(all_clips, axis=0)
    print(f"[DATA] Load hoàn tất. Shape dữ liệu: {final_data.shape}")
    return final_data