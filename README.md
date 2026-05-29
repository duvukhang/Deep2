# Driver Drowsiness Monitoring

Ứng dụng giám sát tài xế bằng webcam theo thời gian thực. Hệ thống phát hiện dấu hiệu buồn ngủ, mất tập trung, tư thế gật/đổ người và gợi ý điểm nghỉ gần nhất bằng OpenStreetMap.

## Dự án làm được gì

- Nhận diện tài xế bằng MediaPipe FaceMesh, hỗ trợ nhiều khuôn mặt trong khung hình.
- Chọn đúng người lái bằng vùng ROI ghế lái, có thể khóa/mở khóa vùng tài xế từ giao diện.
- Tính EAR để theo dõi mắt nhắm, MAR để theo dõi ngáp, và điểm buồn ngủ tổng hợp.
- Ước lượng tư thế đầu bằng pitch/yaw/roll, có baseline theo từng người lái để giảm sai lệch khi camera đặt lệch.
- Nhận diện chuỗi tư thế ngủ gật:
  - Gật đầu/cúi đầu rồi bật lên nhiều lần.
  - Đổ người hoặc chồm người lặp lại.
  - Một lần chồm lấy đồ ngắn vẫn được grace để tránh cảnh báo giả.
- Tự cân sáng camera:
  - Tăng sáng khi thiếu sáng ban đêm.
  - Giảm chói khi nắng gắt.
  - Tăng tương phản khi hình bị nhạt.
- Chuyển chế độ nhận diện theo tình huống:
  - Không che mặt: mắt + miệng + tư thế.
  - Đeo kính râm: miệng + tư thế.
  - Đeo khẩu trang: mắt + tư thế.
  - Camera/góc nhìn kém: giảm cảnh báo giả.
- Cảnh báo âm thanh khi xác nhận buồn ngủ.
- Lưu lịch sử cảnh báo mới nhất trong `data/alert_logs.json`.
- Dashboard web realtime bằng Flask-SocketIO:
  - Xem camera trực tiếp.
  - Xem EAR, MAR, pose, điểm buồn ngủ, số lần gật đầu, số lần đổ/chồm.
  - Xem trạng thái cân sáng và độ tin cậy camera.
- Bản đồ OpenStreetMap:
  - Lấy vị trí hiện tại bằng GPS trình duyệt, fallback sang IP.
  - Tìm điểm nghỉ, cây xăng, quán cà phê, nhà hàng, khách sạn, bãi đỗ xe gần vị trí hiện tại.
  - Ưu tiên điểm nghỉ phía trước hướng di chuyển nếu có heading.
  - Nhập địa chỉ cần đến, tìm tuyến đường và tự dẫn tới điểm nghỉ gần nhất từ vị trí hiện tại.
  - Vẽ đường tới điểm nghỉ bằng OSRM, fallback đường thẳng nếu dịch vụ route lỗi.
- Có cache kết quả OpenStreetMap để giảm gọi API lặp lại.
- Có test tự động cho các service chính.

## Cấu trúc chính

- `app.py`: ứng dụng Flask chính, camera loop, SocketIO, API bản đồ và giao diện web.
- `services/drowsiness_detector.py`: logic phát hiện buồn ngủ, mắt/miệng/tư thế/gật đầu/đổ người.
- `services/camera_lighting.py`: cân sáng, giảm chói và tăng tương phản frame camera.
- `services/driver_selector.py`: chọn đúng vùng tài xế.
- `services/osm_service.py`: tìm điểm nghỉ, geocode địa chỉ và lấy route.
- `services/location_service.py`: lấy vị trí bằng IP khi GPS trình duyệt không khả dụng.
- `services/alert_logger.py`: lưu lịch sử cảnh báo.
- `templates/index.html`: giao diện dashboard.
- `static/js/main.js`: cập nhật realtime, bản đồ, tìm điểm nghỉ và route.
- `static/css/style.css`: giao diện dashboard.
- `tests/test_services.py`: test service.
- `train_yolo.py`, `train_hybrid.py`, `evaluate_hybrid.py`: script train/evaluate mô hình.
- `main.py`: bản thử nghiệm cũ dùng YOLO/hybrid, app hiện tại nên chạy bằng `app.py`.

## Cài đặt

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy ứng dụng

```powershell
venv\Scripts\activate
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:5000
```

Nếu trình duyệt hỏi quyền vị trí, hãy bấm cho phép để bản đồ lấy GPS chính xác hơn. Nếu không cấp quyền, hệ thống sẽ fallback sang vị trí theo IP.

## Cách dùng nhanh

- Đặt webcam nhìn rõ mặt người lái.
- Chỉnh ROI ghế lái nếu camera thấy nhiều người.
- Theo dõi trạng thái lớn trên dashboard:
  - `TỈNH TÁO`: bình thường.
  - `CÓ DẤU HIỆU MỆT`: điểm buồn ngủ bắt đầu tăng.
  - `CẢNH BÁO TƯ THẾ NGỦ GẬT`: gật đầu/đổ người lặp lại.
  - `NGỦ GẬT`: hệ thống xác nhận nguy hiểm.
- Nhấn `Tìm điểm nghỉ ngay` để tìm điểm dừng gần vị trí hiện tại.
- Nhập địa chỉ vào ô bản đồ rồi nhấn `Tìm đường` để hệ thống tìm điểm nghỉ gần nhất trên hành trình.

## Logic tránh cảnh báo giả

- Chồm/cúi lấy đồ một lần ngắn được xem là grace, không cảnh báo ngủ gật ngay.
- Chỉ khi gật đầu hoặc đổ/chồm người lặp lại trong một cửa sổ thời gian ngắn thì mới cộng điểm tư thế.
- Khẩu trang được nhận diện bảo thủ hơn: so sánh vùng miệng với vùng má, tránh nhầm bóng tối/râu/ánh sáng kém là khẩu trang.
- Khi camera quá tối/chói/góc xấu, hệ thống giảm độ tự tin để hạn chế cảnh báo sai.

## Chạy test

```powershell
venv\Scripts\activate
python -m unittest discover -s tests
```

Kiểm tra cú pháp nhanh:

```powershell
python -m py_compile app.py config.py services\camera_lighting.py services\drowsiness_detector.py services\osm_service.py tests\test_services.py
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
venv\Scripts\activate
python train_yolo.py
```

## Train / evaluate hybrid model

Train hybrid:

```powershell
venv\Scripts\activate
python train_hybrid.py
```

Evaluate:

```powershell
venv\Scripts\activate
python evaluate_hybrid.py
```

## Ghi chú

- App chính hiện tại là `app.py`.
- Các API bản đồ dùng dịch vụ công khai của OpenStreetMap/Nominatim/Overpass/OSRM nên cần internet khi tìm địa chỉ, tìm điểm nghỉ hoặc vẽ đường.
- Khi chạy trên Windows, app dùng `cv2.CAP_DSHOW` để giảm lỗi webcam đen với một số máy.
