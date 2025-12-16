from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose, Input, BatchNormalization, TimeDistributed
from tensorflow.keras.models import Model, Sequential

def build_autoencoder(input_shape=(10, 128, 128, 1)):
    """
    Spatiotemporal Autoencoder.
    Input: (Batch, 10, 128, 128, 1) -> Output: (Batch, 10, 128, 128, 1)
    """
    model = Sequential()

    # --- ENCODER (Trích xuất đặc trưng không gian + thời gian) ---
    # Conv3D giúp bắt chuyển động cơ bản
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 2, 2), 
                     padding='same', activation='relu', input_shape=input_shape))
    # Output: (10, 64, 64, 32)
    model.add(BatchNormalization())
    
    # ConvLSTM2D layer 1: Học chuỗi
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True))
    # Output: (10, 64, 64, 16)
    model.add(BatchNormalization())

    # --- DECODER (Tái tạo lại video) ---
    # ConvLSTM2D layer 2: Giải mã chuỗi
    model.add(ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=True))
    # Output: (10, 64, 64, 16)
    model.add(BatchNormalization())
    
    # Conv3DTranspose: Phóng to lại kích thước ảnh (Upsample spatial)
    model.add(Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), strides=(1, 2, 2), 
                              padding='same', activation='relu'))
    # Output: (10, 128, 128, 32)
    
    # Output Layer: Trả về ảnh gốc
    model.add(Conv3DTranspose(filters=1, kernel_size=(3, 3, 3), padding='same', activation='sigmoid'))
    # Output: (10, 128, 128, 1)

    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    model = build_autoencoder()
    model.summary()
    print("[OK] Spatiotemporal Model ready.")