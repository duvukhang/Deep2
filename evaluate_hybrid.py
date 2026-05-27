import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from train_hybrid import SequenceHybridDataset
from models.hybrid_model import DriverMonitoringSystem


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SequenceHybridDataset(data_dir="dataset_extracted", mode="val", seq_len=30)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model = DriverMonitoringSystem(feature_dim=256).to(device)
    model.load_state_dict(torch.load("weights/hybrid_best.pth", map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for spatial_seq, geo_seq, labels in loader:
            spatial_seq = spatial_seq.to(device)
            geo_seq = geo_seq.to(device)

            outputs = model(spatial_seq, geo_seq)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1-score:", f1_score(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()