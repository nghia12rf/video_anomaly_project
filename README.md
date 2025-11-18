# Hệ Thống Phát Hiện Hành Vi Bất Thường Trong Video Giám Sát

## 📌 Giới thiệu
Đồ án môn học: Xây dựng hệ thống phát hiện các sự kiện bất thường trong video giám sát (người đi xe đạp, trượt ván, chạy, phương tiện lạ...) sử dụng kỹ thuật **Deep Learning**.

Dự án áp dụng kiến trúc mạng **Convolutional Autoencoder (CAE)** để học các đặc trưng chuyển động bình thường từ bộ dữ liệu **UCSD Ped2** và cảnh báo khi phát hiện sai số tái tạo (Reconstruction Error) vượt quá ngưỡng cho phép.

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.10
- **Framework:** TensorFlow / Keras
- **Xử lý ảnh:** OpenCV, Imutils
- **Thư viện hỗ trợ:** Numpy, Matplotlib, Scikit-learn

## 📊 Dữ liệu
Sử dụng **UCSD Ped2 Dataset** - bộ dữ liệu tiêu chuẩn cho bài toán phát hiện bất thường trong video, bao gồm:
- 16 video training (chỉ chứa người đi bộ)
- 12 video test (xuất hiện các hành vi bất thường như xe đạp, xe máy, xe đẩy)

## 📂 Cấu trúc dự án
```
video_anomaly_project/
├── data/               # (Thư mục chứa dữ liệu - Cần tạo thủ công)
│   ├── ucsd/
│   │   ├── train/      # Chứa các folder ảnh train (Train001 -> Train016)
│   │   └── test/       # Chứa các folder ảnh test (Test001 -> Test012)
│   └── avenue/         # (Tùy chọn) Chứa file video .avi
├── outputs/            # Nơi lưu Model (.h5), file Ngưỡng (.txt) và Logs
├── src/                # Mã nguồn chính
│   ├── autoencoder.py        # Định nghĩa kiến trúc mạng
│   ├── dataset.py            # Xử lý và load dữ liệu
│   ├── train_autoencoder.py  # Huấn luyện mô hình
│   ├── evaluate.py           # Đánh giá và tính ngưỡng
│   └── realtime_demo.py      # Chạy demo phát hiện
├── requirements.txt    # Danh sách thư viện cần thiết
└── README.md           # Tài liệu hướng dẫn
```

## 🚀 Hướng dẫn cài đặt

### Bước 1: Clone dự án về máy
```bash
git clone https://github.com/nghia12rf/video_anomaly_project.git
cd video_anomaly_project
```

### Bước 2: Cài đặt môi trường & thư viện
```bash
# Khuyến nghị dùng Conda hoặc Virtualenv
pip install -r requirements.txt
```

### Bước 3: Chuẩn bị dữ liệu (Quan trọng)
Do dữ liệu lớn nên không được đưa lên GitHub. Bạn cần tải thủ công:
1. Tải bộ dữ liệu UCSD Anomaly Detection Dataset
2. Giải nén và copy nội dung folder UCSDped2/Train vào `data/ucsd/train`
3. Copy nội dung folder UCSDped2/Test vào `data/ucsd/test` (Lưu ý: Xóa các folder có đuôi `_gt`)

## 📖 Hướng dẫn sử dụng

### 1. Huấn luyện mô hình (Training)
Dạy mô hình học các hành vi bình thường. Model sau khi train sẽ được lưu tại `outputs/models/anomaly_detector.h5`.

```bash
python src/train_autoencoder.py
```

### 2. Tính toán ngưỡng (Thresholding)
Chạy model trên tập train để phân tích sai số và xác định ngưỡng cảnh báo tối ưu. Kết quả lưu tại `outputs/models/threshold.txt`.

```bash
python src/evaluate.py
```

### 3. Chạy Demo (Realtime Detection)
Chạy thử nghiệm trên video test. Hệ thống sẽ hiển thị khung cảnh báo "BẤT THƯỜNG" màu đỏ khi phát hiện sự kiện lạ.

```bash
python src/realtime_demo.py
```

## 📊 Kết quả mong đợi
- **Bình thường:** Khung hình hiển thị chữ xanh, sai số (MSE) thấp
- **Bất thường:** Khi có xe đạp, xe ô tô hoặc người chạy, sai số tăng vọt vượt qua ngưỡng → Hệ thống báo động đỏ

## ❓ Câu hỏi thường gặp

**Q: Lỗi "File not found" khi chạy training?**
A: Kiểm tra đường dẫn thư mục `data/ucsd/train` và `data/ucsd/test` đã được tạo đúng chưa.

**Q: Demo chạy chậm?**
A: Có thể giảm kích thước khung hình hoặc sử dụng GPU để tăng tốc độ xử lý.
```

