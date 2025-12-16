import gradio as gr
import cv2
import numpy as np
import os
from collections import deque
from tensorflow.keras.models import load_model
from moviepy.editor import VideoFileClip, AudioClip

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "st_autoencoder.h5")
THRESHOLD_PATH = os.path.join(BASE_DIR, "outputs", "models", "threshold.txt")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEMP_VIDEO_PATH = os.path.join(OUTPUT_DIR, "temp_gradio.mp4")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "result_final.mp4")

model = None
default_threshold = 0.00045

# ==========================================
# TẢI TÀI NGUYÊN
# ==========================================
def load_resources():
    global model, default_threshold
    if model is None:
        if os.path.exists(MODEL_PATH):
            print(f"[INFO] Đang tải Model từ: {MODEL_PATH}")
            model = load_model(MODEL_PATH, compile=False)
        else:
            print(f"[ERROR] Không tìm thấy file model tại: {MODEL_PATH}")
            
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            try: 
                val = float(f.read().strip())
                default_threshold = val
                print(f"[INFO] Đã tải ngưỡng mặc định: {default_threshold}")
            except: 
                pass

# ==========================================
# XỬ LÝ VIDEO
# ==========================================
def process_video(video_path, threshold_val, amplify_val):
    load_resources()
    
    if video_path is None:
        return None, "⚠️ LỖI", "Chưa tải video lên."
    if model is None:
        return None, "❌ LỖI MODEL", f"Không tìm thấy file model .h5"

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): 
        fps = 24.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(TEMP_VIDEO_PATH, fourcc, fps, (width, height))

    clip_buffer = deque(maxlen=10)
    anomaly_timeline = [] 
    frame_count = 0
    anom_count = 0
    max_error = 0
    
    last_label = "BINH THUONG"
    last_color = (0, 255, 0)
    last_boxes = [] 
    last_mse = 0

    print(f"[INFO] Bắt đầu xử lý... Threshold={threshold_val:.6f}, Amplify={amplify_val}")

    scale_x = width / 64
    scale_y = height / 64

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (64, 64))
        norm = gray_small.astype("float32") / 255.0
        norm = np.expand_dims(norm, axis=-1)
        
        clip_buffer.append(norm)
        
        current_boxes = []
        mse = 0
        is_anomaly = False

        if len(clip_buffer) == 10:
            input_clip = np.array(clip_buffer)
            input_clip = np.expand_dims(input_clip, axis=0)

            reconstructed = model.predict(input_clip, verbose=0)
            diff = np.abs(input_clip - reconstructed)
            
            mse = np.mean(np.square(diff))
            if mse > max_error: 
                max_error = mse

            if mse > threshold_val:
                is_anomaly = True
                
                err_map = diff[0, -1, :, :, 0]
                err_amplified = err_map * amplify_val * 255 
                err_img = np.clip(err_amplified, 0, 255).astype(np.uint8)

                _, thresh_img = cv2.threshold(err_img, 50, 255, cv2.THRESH_BINARY)
                
                pad = 4 
                thresh_img[:pad, :] = 0
                thresh_img[-pad:, :] = 0
                thresh_img[:, :pad] = 0
                thresh_img[:, -pad:] = 0
                thresh_img[0:8, :] = 0 

                kernel = np.ones((3, 3), np.uint8)
                thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
                thresh_img = cv2.dilate(thresh_img, kernel, iterations=2)

                cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for c in cnts:
                    if cv2.contourArea(c) > 10: 
                        x, y, w, h = cv2.boundingRect(c)
                        
                        roi_err = diff[0, -1, y:y+h, x:x+w, 0]
                        local_mse = np.mean(np.square(roi_err)) if roi_err.size > 0 else 0

                        if local_mse > threshold_val * 4.0:
                            final_x = int(x * scale_x)
                            final_y = int(y * scale_y)
                            final_w = int(w * scale_x)
                            final_h = int(h * scale_y)
                            
                            current_boxes.append((final_x, final_y, final_w, final_h))

            if is_anomaly:
                last_label = "BAT THUONG!"
                last_color = (0, 0, 255)
                anom_count += 1
            else:
                last_label = "BINH THUONG"
                last_color = (0, 255, 0)
            
            last_boxes = current_boxes
            last_mse = mse

        anomaly_timeline.append(last_label == "BAT THUONG!")

        cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
        text_content = f"{last_label} | MSE: {last_mse:.5f}"
        cv2.putText(frame, text_content, (15, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2)
        
        for (fx, fy, fw, fh) in last_boxes:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
        
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    final_output_path = TEMP_VIDEO_PATH

    if any(anomaly_timeline):
        try:
            print("[INFO] Đang tạo âm thanh cảnh báo...")
            timeline_arr = np.array(anomaly_timeline)
            
            def make_audio(t):
                t_obj = np.asanyarray(t)
                idxs = (t_obj * fps).astype(int)
                idxs = np.clip(idxs, 0, len(timeline_arr) - 1)
                mask = timeline_arr[idxs]
                if t_obj.ndim == 0:
                    return float(np.sin(2 * np.pi * 1000 * t_obj)) if mask else 0.0
                audio_res = np.zeros_like(t_obj, dtype=np.float32)
                audio_res[mask] = np.sin(2 * np.pi * 1000 * t_obj[mask])
                return audio_res * 0.5
            
            video_clip = VideoFileClip(TEMP_VIDEO_PATH)
            audio_clip = AudioClip(make_audio, duration=video_clip.duration)
            final_video = video_clip.set_audio(audio_clip)
            final_video.write_videofile(OUTPUT_VIDEO_PATH, codec="libx264", audio_codec="aac", logger=None)
            final_output_path = OUTPUT_VIDEO_PATH
        except Exception as e:
            print(f"[WARNING] Không thể tạo âm thanh: {e}")

    ratio = (anom_count / frame_count) * 100 if frame_count > 0 else 0
    badge = "⚠️ PHÁT HIỆN BẤT THƯỜNG" if anom_count > 0 else "✅ AN TOÀN"
    status_md = (
        f"### Kết quả phân tích:\n"
        f"- **Trạng thái:** {badge}\n"
        f"- **Max MSE:** `{max_error:.6f}`\n"
        f"- **Tỷ lệ bất thường:** `{ratio:.1f}%`\n"
        f"- **Số frame bất thường:** `{anom_count}/{frame_count}`"
    )
    return final_output_path, badge, status_md

# ==========================================
# GIAO DIỆN GRADIO 3.x
# ==========================================
load_resources()

with gr.Blocks() as demo:
    gr.Markdown("# 🕵️ HỆ THỐNG GIÁM SÁT AN NINH AI")
    gr.Markdown("**Phát hiện bất thường tự động dựa trên Deep Learning**")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="📤 Tải video lên")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Cấu hình phát hiện")
                threshold_slider = gr.Slider(
                    minimum=0.00001, 
                    maximum=0.002, 
                    value=default_threshold, 
                    step=0.00001, 
                    label="🎯 Ngưỡng nhạy (Threshold)"
                )
                amplify_slider = gr.Slider(
                    minimum=10, 
                    maximum=100, 
                    value=40, 
                    step=5, 
                    label="🔍 Độ khuếch đại (Amplify)"
                )
            
            analyze_btn = gr.Button("🚀 PHÂN TÍCH NGAY")

        with gr.Column():
            video_output = gr.Video(label="📹 Video kết quả")
            status_badge = gr.Label(label="📊 Kết luận")
            result_text = gr.Markdown()

    analyze_btn.click(
        fn=process_video, 
        inputs=[video_input, threshold_slider, amplify_slider], 
        outputs=[video_output, status_badge, result_text]
    )
    
    gr.Markdown("""
    ---
    ### 📌 Hướng dẫn sử dụng:
    1. **Tải video** cần phân tích  
    2. **Điều chỉnh ngưỡng** (threshold)  
    3. **Tăng amplify** nếu muốn nhìn rõ vùng bất thường hơn  
    4. Nhấn **PHÂN TÍCH** và đợi kết quả  

    **Chú thích:**  
    - 🟢 **BÌNH THƯỜNG**  
    - 🔴 **BẤT THƯỜNG**
    """)

if __name__ == "__main__":
    print("[SYSTEM] Đang khởi chạy hệ thống...")
    demo.launch(share=False)
