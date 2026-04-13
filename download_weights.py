from ultralytics.utils.downloads import attempt_download_asset
import os

# Tạo thư mục weights nếu chưa có
if not os.path.exists('weights'):
    os.makedirs('weights')

# Tên file chuẩn của YOLO11 (thường không có chữ 'v')
file_name = 'yolo11n-pose.pt' 

print(f"--- Đang tải {file_name}... ---")
attempt_download_asset(file_name)

# Sau khi tải xong, YOLO thường để ở thư mục gốc, ta move nó vào weights
if os.path.exists(file_name):
    os.rename(file_name, f'weights/{file_name}')
    print(f"--- Đã chuyển file vào weights/{file_name} ---")