import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

from models.hybrid_model import DriverMonitoringSystem

# ==========================================
# 1. DATASET
# ==========================================
class SequenceHybridDataset(Dataset):
    def __init__(self, mode='train', seq_len=30):
        self.samples = 200
        self.seq_len = seq_len

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        spatial_seq = torch.randn(self.seq_len, 512, dtype=torch.float32)
        geo_seq = torch.randn(self.seq_len, 6, dtype=torch.float32)

        label = torch.tensor(idx % 2, dtype=torch.long)

        return spatial_seq, geo_seq, label


# ==========================================
# 2. TRAIN FUNCTION
# ==========================================
def train_brain_node():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Training on: {device} ---\n")

    # ===== CREATE DIR =====
    os.makedirs("weights", exist_ok=True)

    # ===== MODEL =====
    model = DriverMonitoringSystem(feature_dim=256).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        SequenceHybridDataset(),
        batch_size=4,
        shuffle=True,
        num_workers=0,  # fix lỗi Windows
        pin_memory=True if device.type == "cuda" else False
    )

    epochs = 10
    best_loss = float('inf')

    # ======================================
    # TRAIN LOOP
    # ======================================
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for spatial_seq, geo_seq, labels in loop:
            spatial_seq = spatial_seq.to(device)
            geo_seq = geo_seq.to(device)
            labels = labels.to(device)

            # ===== FORWARD =====
            outputs = model(spatial_seq, geo_seq)

            # CHECK SHAPE (anti bug)
            if outputs.shape[-1] != 2:
                raise ValueError(f"❌ Output shape sai: {outputs.shape}, phải là [B, 2]")

            loss = criterion(outputs, labels)

            # ===== BACKWARD =====
            optimizer.zero_grad()
            loss.backward()

            # chống explode gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # ===== SAVE BEST =====
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "weights/hybrid_best.pth")
            print("💾 Saved BEST model")

    # ===== SAVE FINAL =====
    torch.save(model.state_dict(), "weights/hybrid_final.pth")

    print("\n🎯 TRAINING DONE")
    print("Saved:")
    print(" - weights/hybrid_best.pth")
    print(" - weights/hybrid_final.pth")


# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    train_brain_node()