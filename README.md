
#Hệ Thống Phát Hiện Hành Vi Bất Thường Trong Video Giám Sát##📌 Giới thiệuĐồ án môn học: Xây dựng hệ thống phát hiện các sự kiện bất thường trong video giám sát (người đi xe đạp, trượt ván, phương tiện lạ trong khu vực đi bộ...) sử dụng kỹ thuật **Deep Learning**.

Dự án áp dụng kiến trúc mạng **Spatiotemporal Autoencoder** (kết hợp **Conv3D** và **ConvLSTM2D**) để học các đặc trưng không gian và thời gian từ chuỗi frame liên tiếp. Hệ thống sẽ so sánh video tái tạo với video gốc và đưa ra cảnh báo khi sai số (Reconstruction Error) vượt quá ngưỡng cho phép.

##🛠️ Công nghệ sử dụng* **Ngôn ngữ:** Python 3.10
* **Framework:** TensorFlow / Keras
* **Giao diện Web:** Gradio 3.x
* **Xử lý video:** OpenCV, MoviePy
* **Thư viện hỗ trợ:** Numpy, Matplotlib, Imutils

##📊 Dữ liệu & Tiền xử lýSử dụng **UCSD Ped2 Dataset**:

* **Input:** Chuỗi 10 frame liên tiếp (Clip length = 10).
* **Preprocessing:** Resize về **64x64**, Grayscale, Normalize [0, 1].
* **Dữ liệu:**
* **Train:** 16 video (chỉ chứa người đi bộ - hành vi bình thường).
* **Test:** 12 video (xuất hiện xe đạp, xe ô tô, trượt ván - hành vi bất thường).



##📂 Cấu trúc dự án```text
video_anomaly_project/
├── data/                       # Thư mục chứa dữ liệu
│   ├── ucsd/
│   │   ├── train/              # Folder ảnh train (Train001 -> Train016)
│   │   └── test/               # Folder ảnh test (Test001 -> Test012)
│   └── ucsd_videos/            # (Tự động tạo) Chứa video .mp4 sau khi convert
├── outputs/                    # Nơi lưu kết quả
│   ├── models/                 # Lưu st_autoencoder.h5, threshold.txt
│   ├── logs/                   # Biểu đồ training, histogram lỗi
│   └── videos/                 # Video kết quả từ Gradio
├── src/                        # Mã nguồn chính
│   ├── autoencoder.py          # Kiến trúc Spatiotemporal (Conv3D + ConvLSTM)
│   ├── dataset.py              # Load dữ liệu, tạo sliding window clips
│   ├── train_autoencoder.py    # Huấn luyện mô hình
│   ├── evaluate.py             # Tính toán và lưu ngưỡng (Threshold)
│   ├── realtime_demo.py        # Chạy demo trực tiếp (OpenCV)
│   ├── gradio_app.py           # Giao diện Web (Gradio)
│   └── optical_flow.py         # Module hỗ trợ tính toán Optical Flow
├── tools/                      # Các công cụ hỗ trợ
│   └── convert_ucsd.py         # Tool chuyển dataset ảnh .tif sang video .mp4
├── requirements.txt            # Danh sách thư viện
└── README.md                   # Tài liệu hướng dẫn

```

##🚀 Hướng dẫn cài đặt###Bước 1: Clone dự án```bash
git clone https://github.com/nghia12rf/video_anomaly_project.git
cd video_anomaly_project

```

###Bước 2: Cài đặt thư viện```bash
pip install -r requirements.txt


```

###Bước 3: Chuẩn bị dữ liệu1. Tải bộ dữ liệu **UCSD Ped2**.
2. Copy nội dung folder `Train` vào `data/ucsd/train`.
3. Copy nội dung folder `Test` vào `data/ucsd/test`.

*(Khuyến nghị)* Chạy công cụ trong thư mục `tools` để chuyển đổi ảnh dataset sang video mp4 (cần thiết nếu muốn test nhanh trên Gradio):

```bash
python tools/convert_ucsd.py

```

##📖 Hướng dẫn sử dụng###1. Huấn luyện mô hình (Training)Dạy mô hình học đặc trưng bình thường (Input size 64x64). Model sau khi train sẽ lưu tại `outputs/models/st_autoencoder.h5`.

```bash
python src/train_autoencoder.py

```

###2. Tính toán ngưỡng (Thresholding)Chạy đánh giá trên tập train để tìm ngưỡng (Threshold) tối ưu dựa trên phân phối lỗi (MSE).

```bash
python src/evaluate.py

```

###3. Chạy Demo Trực Tiếp (Realtime OpenCV)Chạy trên dữ liệu test (dạng ảnh hoặc video) có sẵn trong máy.

* **Tính năng:**
* Thanh trượt điều chỉnh độ nhạy (Threshold) và độ hiển thị lỗi (Amplify).
* Vẽ bounding box quanh vùng bất thường.
* Phát âm thanh cảnh báo (Beep) khi phát hiện sự kiện lạ.



```bash
python src/realtime_demo.py

```

###4. Chạy Giao Diện Web (Gradio App)Khởi chạy Web App để upload video bất kỳ và nhận kết quả phân tích.

* **Tính năng:**
* Upload video từ máy tính.
* Tùy chỉnh Threshold/Amplify trực quan.
* Xuất video kết quả kèm **âm thanh báo động** tại các phân đoạn bất thường.
* Hiển thị báo cáo thống kê (Tỷ lệ bất thường, Max MSE).



```bash
python src/gradio_app.py

```



##📊 Kết quả mong đợi* **Trạng thái Bình thường:** Khung hình hiển thị nhãn xanh `BÌNH THƯỜNG`, chỉ số MSE thấp.
* **Trạng thái Bất thường:**
* Khi xuất hiện đối tượng lạ (xe cộ) hoặc hành vi lạ (chạy nhanh).
* MSE tăng vọt vượt ngưỡng.
* Hệ thống khoanh vùng đỏ và phát cảnh báo.



