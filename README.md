
---

# Hệ Thống Phát Hiện Hành Vi Bất Thường Trong Video Giám Sát

## 📌 Giới thiệu
Đồ án môn học: Xây dựng hệ thống phát hiện các sự kiện bất thường trong video giám sát (người đi xe đạp, trượt ván, phương tiện lạ trong khu vực đi bộ...) sử dụng kỹ thuật **Deep Learning**.

Dự án áp dụng kiến trúc **Spatiotemporal Autoencoder** (kết hợp **Conv3D** và **ConvLSTM2D**) để học đặc trưng không gian – thời gian từ chuỗi frame liên tiếp. Hệ thống so sánh video tái tạo với video gốc và cảnh báo khi sai số (Reconstruction Error) vượt ngưỡng.

---

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.10  
- **Framework:** TensorFlow / Keras  
- **Giao diện Web:** Gradio 3.x  
- **Xử lý video:** OpenCV, MoviePy  
- **Thư viện hỗ trợ:** Numpy, Matplotlib, Imutils  

---

## 📊 Dữ liệu & Tiền xử lý
Sử dụng **UCSD Ped2 Dataset**:

- **Input:** Chuỗi 10 frame liên tiếp (Clip length = 10)  
- **Preprocessing:** Resize 64×64, Grayscale, Normalize [0,1]  
- **Train:** 16 video (hành vi bình thường)  
- **Test:** 12 video (có xe đạp, xe ô tô, trượt ván – bất thường)  

---

## 📂 Cấu trúc dự án
```text
video_anomaly_project/
├── data/
│   ├── ucsd/
│   │   ├── train/
│   │   └── test/
│   └── ucsd_videos/
├── outputs/
│   ├── models/
│   ├── logs/
│   └── videos/
├── src/
│   ├── autoencoder.py
│   ├── dataset.py
│   ├── train_autoencoder.py
│   ├── evaluate.py
│   ├── realtime_demo.py
│   ├── gradio_app.py
│   └── optical_flow.py
├── tools/
│   └── convert_ucsd.py
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt

### **Bước 1: Clone dự án**
```bash
git clone https://github.com/nghia12rf/video_anomaly_project.git
cd video_anomaly_project
```

### **Bước 2: Cài đặt thư viện**
```bash
pip install -r requirements.txt
```

### **Bước 3: Chuẩn bị dữ liệu**
1. Tải bộ dữ liệu **UCSD Ped2**  
2. Copy folder `Train` → `data/ucsd/train`  
3. Copy folder `Test` → `data/ucsd/test`  

**(Khuyến nghị)** Chuyển ảnh `.tif` sang video `.mp4` để test nhanh trên Gradio:

```bash
python tools/convert_ucsd.py
```

---

## 📖 Hướng dẫn sử dụng

### **1. Huấn luyện mô hình**
Model lưu tại `outputs/models/st_autoencoder.h5`.

```bash
python src/train_autoencoder.py
```

### **2. Tính toán ngưỡng (Threshold)**
```bash
python src/evaluate.py
```

### **3. Chạy Demo Trực Tiếp (OpenCV)**
- Điều chỉnh Threshold  
- Vẽ bounding box  
- Phát âm thanh cảnh báo  

```bash
python src/realtime_demo.py
```

### **4. Chạy giao diện Web (Gradio)**
- Upload video  
- Điều chỉnh Threshold/Amplify  
- Xuất video kết quả + âm thanh cảnh báo  
- Hiển thị thống kê  

```bash
python src/gradio_app.py
```

---

## 📊 Kết quả mong đợi
- **Bình thường:** Nhãn xanh *BÌNH THƯỜNG*, MSE thấp  
- **Bất thường:**  
  - Xuất hiện xe, trượt ván, chạy nhanh  
  - MSE tăng mạnh  
  - Khoanh vùng đỏ + cảnh báo âm thanh  

---

