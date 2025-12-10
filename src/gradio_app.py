import gradio as gr
import cv2
import numpy as np
import os
import winsound  
from tensorflow.keras.models import load_model
from moviepy.editor import VideoFileClip, AudioClip

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Các đường dẫn Input
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "anomaly_detector.h5")
THRESHOLD_PATH = os.path.join(BASE_DIR, "outputs", "models", "threshold.txt")

# --- CẤU HÌNH OUTPUT  ---
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Đường dẫn file tạm và file kết quả nằm trong thư mục này
TEMP_VIDEO_PATH = os.path.join(OUTPUT_DIR, "temp_video_silent.mp4") 
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "result_final.mp4")

# Biến toàn cục
model = None
default_threshold = 0.0035

def load_resources():
    global model, default_threshold
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, compile=False)
        else:
            if os.path.exists("anomaly_detector.h5"):
                model = load_model("anomaly_detector.h5", compile=False)
    
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            try: default_threshold = float(f.read().strip())
            except: pass

def process_video(video_path, threshold_value):
    load_resources()
    if video_path is None: 
        return None, "LỖI", "Vui lòng upload video!"

    cap = cv2.VideoCapture(video_path)
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 24.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # Lưu vào đường dẫn mới trong outputs/videos
    out = cv2.VideoWriter(TEMP_VIDEO_PATH, fourcc, fps, (width, height))

    max_error = 0
    frame_count = 0
    anom_count = 0
    anomaly_timeline = []

    # Cấu hình Frame Skipping (Tăng tốc)
    SKIP_FRAMES = 2
    last_label = "BINH THUONG"
    last_color = (0, 255, 0)
    last_mse = 0
    last_boxes = [] 
    scale_x = width / 128
    scale_y = height / 128

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # CHỈ CHẠY MODEL KHI ĐẾN LƯỢT
        if frame_count % (SKIP_FRAMES + 1) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_resized = cv2.resize(gray, (128, 128))
            input_data = gray_resized.astype("float32") / 255.0
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.expand_dims(input_data, axis=-1)

            reconstructed = model.predict(input_data, verbose=0)
            
            diff = np.abs(input_data - reconstructed)
            mse = np.mean(np.square(diff))
            if mse > max_error: max_error = mse

            if mse > threshold_value:
                last_label = "CANH BAO!"
                last_color = (0, 0, 255) # Đỏ
                anom_count += 1 
                last_mse = mse
                
                # Tìm khung vẽ
                last_boxes = [] 
                diff_map = (diff[0, :, :, 0] * 255).astype(np.uint8)
                _, thresh_img = cv2.threshold(diff_map, 30, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    if cv2.contourArea(c) > 10:
                        x, y, w, h = cv2.boundingRect(c)
                        last_boxes.append((int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)))
            else:
                last_label = "BINH THUONG"
                last_color = (0, 255, 0)
                last_mse = mse
                last_boxes = []

        else:
            if last_label == "CANH BAO!":
                anom_count += 1

        anomaly_timeline.append(last_label == "CANH BAO!")

        # Vẽ lên frame
        for (x, y, w, h) in last_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"{last_label} | MSE: {last_mse:.4f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 10 == 0 and last_label == "CANH BAO!":
            try: winsound.Beep(1000, 50)
            except: pass

    cap.release()
    out.release()

    # --- HẬU KỲ: CHÈN ÂM THANH ---
    if len(anomaly_timeline) > 0:
        print("[INFO] Đang render âm thanh...")
        timeline_arr = np.array(anomaly_timeline)

        def make_audio(t):
            t_obj = np.asanyarray(t)
            idxs = (t_obj * fps).astype(int)
            idxs = np.clip(idxs, 0, len(timeline_arr) - 1)
            mask = timeline_arr[idxs]
            
            if t_obj.ndim == 0:
                if mask: return float(np.sin(2 * np.pi * 800 * t_obj))
                return 0.0
            
            audio_res = np.zeros_like(t_obj, dtype=np.float32)
            audio_res[mask] = np.sin(2 * np.pi * 800 * t_obj[mask])
            return audio_res

        try:
            video_clip = VideoFileClip(TEMP_VIDEO_PATH)
            audio_clip = AudioClip(make_audio, duration=video_clip.duration)
            final_video = video_clip.set_audio(audio_clip)
            # Lưu vào outputs/videos
            final_video.write_videofile(OUTPUT_VIDEO_PATH, codec="libx264", audio_codec="aac", logger=None)
            return_video = OUTPUT_VIDEO_PATH
        except Exception as e:
            print(f"Lỗi render âm thanh: {e}")
            return_video = TEMP_VIDEO_PATH
    else:
        return_video = TEMP_VIDEO_PATH

    # Báo cáo
    ratio = (anom_count / frame_count) * 100 if frame_count > 0 else 0
    if anom_count > 0:
        badge = "⚠️ CẢNH BÁO"
        status = (f"### Kết quả phân tích:\n"
                  f"- **Trạng thái:** Bất thường (Có khoanh vùng lỗi).\n"
                  f"- **Sai số Max:** `{max_error:.5f}`\n"
                  f"- **Tỉ lệ lỗi:** `{ratio:.1f}%`\n"
                  f"- **Lưu tại:** `{OUTPUT_VIDEO_PATH}`")
    else:
        badge = "✅ AN TOÀN"
        status = f"Bình thường. MSE Max: `{max_error:.5f}`"

    return return_video, badge, status

# --- GIAO DIỆN ---
load_resources()
with gr.Blocks(title="Hệ thống giám sát thông minh") as demo:
    gr.Markdown("<h1 style='text-align: center'>🕵️ HỆ THỐNG GIÁM SÁT (OUTPUTS FOLDER)</h1>")
    
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input", sources=["upload"])
            slider = gr.Slider(0.001, 0.01, default_threshold, step=0.0001, label="Ngưỡng")
            btn = gr.Button("🚀 PHÂN TÍCH", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Output")
            badge = gr.Label()
            text = gr.Markdown()

    btn.click(process_video, [video_in, slider], [video_out, badge, text])

if __name__ == "__main__":
    demo.launch()