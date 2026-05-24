from ultralytics import YOLO
import torch
import os

def traint_spatail_node():
    print("\n Start train yolo v2 (Anti-Overfitting)\n")
    
    #== Check Device ==
    device=0 if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    #== Path ==
    #Su dung yolo11
    weights_path = 'weights/yolo11n.pt'
    data_path='configs/data.yaml'
    
    #== Check Data ==
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Khong tim thay file: {data_path}")

    #== Create Dir ==
    os.makedirs("weights", exist_ok=True)
    
    #== Load Model ==
    if os.path.exists(weights_path):
        print(f"Load local weights: {weights_path}")
        model = YOLO(weights_path)
    else:
        print("Dowload yolo11n.pt tu Ultralytics...")
        model=YOLO('yolo11n.pt')
    
    #== Train ==
    results=model.train(
        data=data_path,
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        project='runs/detect',
        name='yolo_drowsy_v2',
        
        #== them chi so chong overfitting ==
        patience=20,
        dropout=0.1,
        degrees=15.0,
        mosaic=1.0,
        close_mosaic=10
    )
    print("\n Hoan tat train yolo!")
    
if __name__ == '__main__':
    traint_spatail_node()
    