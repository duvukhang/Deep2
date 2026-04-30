from ultralytics import YOLO
import torch
import os

def train_spatial_node():
    print("\n🚀 START TRAIN YOLO\n")

    # ===== CHECK DEVICE =====
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"👉 Device: {device}")

    # ===== PATH =====
    weights_path = 'weights/yolo8.pt'
    data_path = 'configs/data.yaml'

    # ===== CHECK DATA =====
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {data_path}")

    # ===== CREATE DIR =====
    os.makedirs("weights", exist_ok=True)

    # ===== LOAD MODEL =====
    if os.path.exists(weights_path):
        print(f"✅ Load local weights: {weights_path}")
        model = YOLO(weights_path)
    else:
        print("⬇️ Download yolo11n.pt từ Ultralytics...")
        model = YOLO('yolo11n.pt')

    # ===== TRAIN =====
    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=4,
        device=device,
        workers=0,          # 🔥 fix lỗi Windows
        project='runs/detect',
        name='yolo_drowsy_v1',
        exist_ok=True,      # không lỗi khi trùng tên
        pretrained=True,
        verbose=True,
        cache=True          # tăng tốc load dataset
    )

    print("\n🎯 TRAIN DONE")
    print("👉 Best weights:", os.path.join('runs/detect/yolo_drowsy_v1', 'weights/best.pt'))


if __name__ == "__main__":
    train_spatial_node()