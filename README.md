# Driver Drowsiness Monitoring

Ứng dụng giám sát tài xế bằng webcam, phát hiện dấu hiệu buồn ngủ và gợi ý điểm nghỉ gần vị trí hiện tại.

## Chức năng chính

- Nhận diện tài xế bằng MediaPipe FaceMesh.
- Chuyển chế độ theo tình huống:
  - Không che mặt: mắt + miệng + tư thế.
  - Đeo kính râm: miệng + tư thế.
  - Đeo khẩu trang: mắt + tư thế.
- Lưu 30 cảnh báo mới nhất trong `data/alert_logs.json`.
- Tìm điểm nghỉ bằng OpenStreetMap, có cache và ưu tiên hướng di chuyển nếu GPS trả về heading.

## Cài đặt

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy ứng dụng

```powershell
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

## Train YOLO

Dataset YOLO được cấu hình ở `configs/data.yaml` bằng path tương đối:

```yaml
path: driver_drowsiness_Computer_Vision_Model
train: train/images
val: valid/images
test: test/images
```

Train:

```powershell
python train_yolo.py
```
