import cv2
import numpy as np

def draw_optical_flow(frame, prev_frame, step=16):
    """
    Tính toán và vẽ Optical Flow (Dense) bằng thuật toán Farneback.
    - frame: Ảnh hiện tại (Grayscale hoặc BGR)
    - prev_frame: Ảnh trước đó (Grayscale)
    """
    if prev_frame is None:
        return frame
    
    # Đảm bảo đầu vào là ảnh xám
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
    else:
        gray = frame
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
    # Tính Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 
                                        pyr_scale=0.5, levels=3, winsize=15, 
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # Vẽ các đường chuyển động (chỉ vẽ lưới thưa để dễ nhìn)
    h, w = gray.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    
    # Tạo các đường line xanh lá cây
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    
    # Vẽ chấm đỏ tại điểm bắt đầu
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        
    return vis

# --- Test Block ---
if __name__ == "__main__":
    print("Function draw_optical_flow ready.")