# Video Anomaly Detection Project

This project detects abnormal behaviors (running, falling, crowding) in surveillance videos using:

- Optical Flow (motion extraction)
- Autoencoder (unsupervised anomaly detection)
- PyTorch for training
- OpenCV for realtime alert system

## Folder Structure
```
video_anomaly_project/
│
├── data/               # Place UCSD/Avenue dataset here (not included)
│   ├── ucsd/
│   └── avenue/
│
├── src/                # Source code
│   ├── dataset.py
│   ├── optical_flow.py
│   ├── autoencoder.py
│   ├── train_autoencoder.py
│   ├── evaluate.py
│   └── realtime_demo.py
│
├── outputs/            # Models & logs will be stored here
│   ├── models/
│   └── logs/
│
├── notebooks/          # Optional Jupyter notebooks
├── requirements.txt    # Required Python libraries
└── README.md           # Project documentation
```

## How to Install
```
conda create -n video_anomaly python=3.10 -y
conda activate video_anomaly
pip install -r requirements.txt
```

## How to Run Training
```
python src/train_autoencoder.py
```

## How to Run Evaluation
```
python src/evaluate.py
```

## How to Run Realtime Demo
```
python src/realtime_demo.py
```

## Notes
- Put UCSD/Avenue dataset into `data/` folder manually.
- Make sure your webcam works for realtime demo.
