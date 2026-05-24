import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm

from models.hybrid_model import DriverMonitoringSystem

# ==========================================
# 1. DATASET
# ==========================================
class SequenceHybridDataset(Dataset):
    def __init__(self, data_dir="dataset_extracted", mode='train', seq_len=30):
        """
        Đọc dữ liệu đã trích xuất đặc trưng thay vì dùng Dummy Data.
        Yêu cầu thư mục cấu trúc: dataset_extracted/train và dataset_extracted/val
        """
        self.seq_len = seq_len
        self.mode = mode
        self.data_files = []
        
        # Đường dẫn thư mục theo mode (train/val)
        mode_dir = os.path.join(data_dir, mode)
        if os.path.exists(mode_dir):
            self.data_files = [os.path.join(mode_dir, f) for f in os.listdir(mode_dir) if f.endswith('.npy')]

        # Cảnh báo nếu chưa có dữ liệu thật
        self.use_dummy = len(self.data_files) == 0
        if self.use_dummy:
            print(f"⚠️ CẢNH BÁO: Không có dữ liệu thật tại '{mode_dir}'. Đang dùng Dummy Data để test luồng!")
            self.samples = 150 if mode == 'train' else 30

    def __len__(self):
        return len(self.data_files) if not self.use_dummy else self.samples

    def __getitem__(self, idx):
        # 1. Trả về dữ liệu giả nếu test
        if self.use_dummy:
            spatial_seq = torch.randn(self.seq_len, 512, dtype=torch.float32)
            geo_seq = torch.randn(self.seq_len, 6, dtype=torch.float32)
            label = torch.tensor(idx % 2, dtype=torch.long)
            return spatial_seq, geo_seq, label
        
        # 2. Đọc dữ liệu thật (Bạn cấu trúc file .npy lưu dạng Dictionary)
        # data = {'spatial': [...], 'geo': [...], 'label': 1}
        data = np.load(self.data_files[idx], allow_pickle=True).item()
        spatial_seq = torch.tensor(data['spatial'], dtype=torch.float32)
        geo_seq = torch.tensor(data['geo'], dtype=torch.float32)
        label = torch.tensor(data['label'], dtype=torch.long)
        
        return spatial_seq, geo_seq, label

# ==========================================
# 2. TRAIN FUNCTION WITH VALIDATION
# ==========================================
def train_brain_node():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training Hybrid on: {device} ---\n")

    os.makedirs("weights", exist_ok=True)

    # ===== MODEL =====
    model = DriverMonitoringSystem(feature_dim=256).to(device)

    # Thêm weight_decay (L2 Regularization) để phạt weights quá lớn
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # ===== DATALOADER =====
    train_loader = DataLoader(SequenceHybridDataset(mode='train'), batch_size=8, shuffle=True)
    val_loader = DataLoader(SequenceHybridDataset(mode='val'), batch_size=8, shuffle=False)

    epochs = 50
    best_val_loss = float('inf')
    patience = 0
    patience_limit = 10 # Early stopping cho mạng Hybrid

    for epoch in range(epochs):
        # --- PHẦN 1: HUẤN LUYỆN (TRAIN) ---
        model.train()
        total_train_loss = 0
        train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")

        for spatial_seq, geo_seq, labels in train_loop:
            spatial_seq, geo_seq, labels = spatial_seq.to(device), geo_seq.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(spatial_seq, geo_seq)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Chống nổ gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # --- PHẦN 2: KIỂM THỬ (VALIDATION) ---
        model.eval()
        total_val_loss = 0
        val_loop = tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{epochs}")

        with torch.no_grad(): # Tắt tính toán đạo hàm để tiết kiệm RAM
            for spatial_seq, geo_seq, labels in val_loop:
                spatial_seq, geo_seq, labels = spatial_seq.to(device), geo_seq.to(device), labels.to(device)
                outputs = model(spatial_seq, geo_seq)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"✅ Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- ĐÁNH GIÁ LƯU MODEL VÀ DỪNG SỚM ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), "weights/hybrid_best.pth")
            print("💾 Đã lưu BEST model (Val Loss giảm)")
        else:
            patience += 1
            print(f"⚠️ Val Loss không giảm. Patience: {patience}/{patience_limit}")
            if patience >= patience_limit:
                print("🛑 KÍCH HOẠT EARLY STOPPING: Đã dừng huấn luyện để chống Overfitting!")
                break

    # Lưu trọng số cuối cùng
    torch.save(model.state_dict(), "weights/hybrid_final.pth")
    print("\n🎯 TRAINING DONE")

if __name__ == '__main__':
    train_brain_node()