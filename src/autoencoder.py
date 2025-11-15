from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np

def build_autoencoder(input_shape=(128, 128, 1)):
    """
    Xây dựng Convolutional Autoencoder.
    Input: (128, 128, 1) -> Output: (128, 128, 1)
    """
    # --- ENCODER (Nén dữ liệu) ---
    input_img = Input(shape=input_shape)
    
    # Block 1: 128 -> 64
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Block 2: 64 -> 32
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # --- DECODER (Tái tạo dữ liệu) ---
    
    # Block 3: 32 -> 64
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    # Block 4: 64 -> 128
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    
    # Output Layer: Trả về ảnh gốc (dùng Sigmoid để giá trị về 0-1)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Tạo Model
    autoencoder = Model(input_img, decoded)
    
    # Compile Model (Dùng MSE Loss để đo độ sai lệch giữa ảnh gốc và ảnh tái tạo)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# --- Test Block ---
if __name__ == "__main__":
    print("--- ĐANG KIỂM TRA KIẾN TRÚC MODEL ---")
    try:
        model = build_autoencoder()
        model.summary() # In ra cấu trúc mạng
        print("\n[OK] Model đã được xây dựng thành công!")
    except Exception as e:
        print(f"\n[ERROR] Lỗi xây dựng model: {e}")
        print("Gợi ý: Kiểm tra xem đã cài tensorflow chưa (pip install tensorflow)")