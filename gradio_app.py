import os
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from dataset import load_video_file

# cấu hình 
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "anomaly_detector.h5")
THRESHOLD_PATH = os.path.join(BASE_DIR, "outputs", "models", "threshold.txt")

model = None
default_threshold = 0.0035  # fallback nếu không tìm thấy file

# load model, threshold

def load_resources():
    global model, default_threshold

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
        model = load_model(MODEL_PATH, compile=False)

    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            default_threshold = float(f.read().strip())


#hàm phân tích gradio
def analyze_video_ui(video_path, threshold_value):
    """
    Trả về:
      - video_path: để hiển thị lại video
      - badge: CẢNH BÁO hay BÌNH THƯỜNG
      - status_text: Markdown mô tả trạng thái + MSE
    """
    load_resources()

    frames = load_video_file(video_path, resize=(128, 128))
    if frames is None or len(frames) == 0:
        return None, "LỖI", "Không đọc được video hoặc không trích được frame!"

    if frames.ndim == 3:
        frames = np.expand_dims(frames, axis=-1)

    reconstructed = model.predict(frames, verbose=0)
    mse = np.mean(np.square(frames - reconstructed), axis=(1, 2, 3))

    max_err = float(np.max(mse))
    num_anom = int(np.sum(mse > threshold_value))
    ratio = num_anom / len(mse)

# kết luận
    if max_err > threshold_value:
        badge = "⚠ CẢNH BÁO: BẤT THƯỜNG"
        status_text = (
            f"**Trạng thái:** Phát hiện hành vi bất thường.\n\n"
            f"- Sai số lớn nhất (MSE): **{max_err:.6f}** (vượt ngưỡng {threshold_value:.4f})\n"
            f"- Số frame bất thường: **{num_anom}/{len(mse)}** (~{ratio*100:.1f}%)"
        )
    else:
        badge = "BÌNH THƯỜNG"
        status_text = (
            f"**Trạng thái:** Không phát hiện bất thường.\n\n"
            f"- Sai số lớn nhất (MSE): **{max_err:.6f}** (≤ ngưỡng {threshold_value:.4f})"
        )

    return video_path, badge, status_text


#gradio
with gr.Blocks() as demo:

    load_resources()
#header
    gr.Markdown(
        """
        <h1 style="text-align:center; color:#e67e22;">
            HỆ THỐNG PHÁT HIỆN HÀNH VI BẤT THƯỜNG
        </h1>
        """
    )

    with gr.Row():
        #input
        with gr.Column(scale=1):
            gr.Markdown("### Đầu vào (Input)")

            video_in = gr.Video(label="Kéo thả video vào đây hoặc nhấn để tải lên")

            threshold_slider = gr.Slider(
                minimum=0.0005,
                maximum=0.01,
                value=default_threshold,
                step=0.0001,
                label="Độ nhạy (Threshold)",
                info="Kéo trái để nhạy hơn (dễ báo giả), kéo phải để ít nhạy hơn."
            )

            analyze_btn = gr.Button("BẮT ĐẦU PHÂN TÍCH", variant="primary")

        #output
        with gr.Column(scale=1):
            gr.Markdown("### Kết quả (Output)")

            video_out = gr.Video(label="")
            badge_out = gr.Label(label="Trạng thái tổng quát")
            status_md = gr.Markdown()

    analyze_btn.click(
        fn=analyze_video_ui,
        inputs=[video_in, threshold_slider],
        outputs=[video_out, badge_out, status_md]
    )

if __name__ == "__main__":
    demo.launch()
