import os
# Tắt log rác của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import glob
from collections import deque
from tensorflow.keras.models import load_model

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "st_autoencoder.h5")
THRESHOLD_PATH = os.path.join(BASE_DIR, "outputs", "models", "threshold.txt")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "ucsd", "test", "Test004") 

# [THÊM] Hàm phát âm thanh an toàn (chỉ chạy trên Windows)
def play_sound():
    try:
        import winsound
        # Tần số 1000Hz, độ dài 100ms (ngắn để không bị giật lag video)
        winsound.Beep(1000, 100) 
    except ImportError:
        pass 

def nothing(x):
    pass

def main():
    # 1. Load Ngưỡng
    threshold = 0.0005 
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            try: threshold = float(f.read())
            except: pass
    print(f"Ngưỡng gốc từ file: {threshold:.8f}")

    print("Loading model...")
    if not os.path.exists(MODEL_PATH):
        print("Lỗi: Không tìm thấy model!")
        return
    model = load_model(MODEL_PATH, compile=False)

    frames_list = []
    is_video_file = False
    if os.path.isdir(TEST_DATA_PATH):
        frames_list = sorted(glob.glob(os.path.join(TEST_DATA_PATH, "*.tif")) + 
                             glob.glob(os.path.join(TEST_DATA_PATH, "*.jpg")))
    else:
        is_video_file = True
        cap = cv2.VideoCapture(TEST_DATA_PATH)

    cv2.namedWindow("Spatiotemporal Detection")
    cv2.namedWindow("Difference Map (Amplified)") 
    
    # Thanh trượt độ mịn cao (1 triệu đơn vị)
    scale_factor = 1000000
    init_val = int(threshold * scale_factor)
    
    cv2.createTrackbar("Threshold (x1M)", "Spatiotemporal Detection", init_val, 1000, nothing)
    cv2.createTrackbar("Amplify", "Difference Map (Amplified)", 30, 100, nothing)

    clip_buffer = deque(maxlen=10) 
    
    idx = 0
    while True:
        if is_video_file:
            ret, frame = cap.read()
            if not ret: break
        else:
            if idx >= len(frames_list): break
            frame = cv2.imread(frames_list[idx])
            idx += 1
        
        if frame is None: break

        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (64, 64))
        
        norm = gray_small.astype("float32") / 255.0
        norm = np.expand_dims(norm, axis=-1) 

        clip_buffer.append(norm)

        label = "Dang thu thap..."
        color = (255, 255, 0)
        mse = 0
        boxes = []
        is_alarm = False
        debug_final = np.zeros((200, 400), dtype=np.uint8)

        trackbar_val = cv2.getTrackbarPos("Threshold (x1M)", "Spatiotemporal Detection")
        amp_val = cv2.getTrackbarPos("Amplify", "Difference Map (Amplified)")
        if amp_val < 1: amp_val = 1
        
        current_threshold = trackbar_val / float(scale_factor) if trackbar_val > 0 else threshold

        if len(clip_buffer) == 10:
            input_clip = np.array(clip_buffer)
            input_clip = np.expand_dims(input_clip, axis=0)
            
            reconstructed = model.predict(input_clip, verbose=0)
            diff = np.abs(input_clip - reconstructed)
            mse = np.mean(np.square(diff)) 
            
            # --- Xử lý ảnh lỗi ---
            err_map = diff[0, -1, :, :, 0] 
            err_amplified = err_map * amp_val * 255
            err_img = np.clip(err_amplified, 0, 255).astype(np.uint8)
            
            _, thresh_img = cv2.threshold(err_img, 50, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((3, 3), np.uint8)
            thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh_img = cv2.dilate(thresh_img, kernel, iterations=2)
            
            debug_img = cv2.resize(err_img, (200, 200), interpolation=cv2.INTER_NEAREST)
            debug_thresh = cv2.resize(thresh_img, (200, 200), interpolation=cv2.INTER_NEAREST)
            debug_final = cv2.hconcat([debug_img, debug_thresh])

            cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scale_x = display_frame.shape[1] / 64
            scale_y = display_frame.shape[0] / 64
            
            for c in cnts:
                if cv2.contourArea(c) > 15: 
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append((int(x * scale_x), int(y * scale_y), 
                                  int(w * scale_x), int(h * scale_y)))

            if mse > current_threshold:
                label = "BAT THUONG!"
                color = (0, 0, 255)
                is_alarm = True
                
                # [THÊM] Gọi hàm phát âm thanh khi có báo động
                play_sound() 
            else:
                label = "Binh thuong"
                color = (0, 255, 0)

        # Draw info
        cv2.putText(display_frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display_frame, f"MSE: {mse:.6f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(display_frame, f"Thresh: {current_threshold:.6f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for (x, y, w, h) in boxes:
            box_color = (0, 0, 255) if is_alarm else (0, 255, 255)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)

        cv2.imshow("Spatiotemporal Detection", display_frame)
        cv2.imshow("Difference Map (Amplified)", debug_final) 
        
        if cv2.waitKey(30) & 0xFF == ord('q'): break

    if is_video_file: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()