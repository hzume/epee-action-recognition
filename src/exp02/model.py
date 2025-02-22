import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import json
from torch.utils.data import DataLoader

from dataset import Dataset1, get_max_size

class Model1(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet34d", pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=5)
        self.neck = nn.LSTM(self.backbone.num_features, 128, 1, batch_first=True)
        self.head = nn.Linear(128, num_classes)
        self.dropout_backbone = nn.Dropout(dropout)
        self.dropout_neck = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, num_frames, height, width, channels)
        b, n, h, w, c = x.shape
        x = x.reshape(b * n, h, w, c).permute(0, 3, 1, 2) # (b*n, c, h, w)

        y_coords = torch.linspace(0, 1, h).to(x.device)
        x_coords = torch.linspace(0, 1, w).to(x.device)
        
        y_coords = y_coords.unsqueeze(1).repeat(1, w) # (h, w)
        x_coords = x_coords.unsqueeze(0).repeat(h, 1) # (h, w)

        y_coords = y_coords.unsqueeze(0).unsqueeze(0).repeat(b*n, 1, 1, 1) 
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).repeat(b*n, 1, 1, 1)


        x_left = x 
        x_left = torch.cat([x_left, y_coords, x_coords], dim=1)

        x_right = x.clone().flip(2)
        x_right = torch.cat([x_right, y_coords, x_coords], dim=1)

        x_left = self.backbone(x_left)
        x_left = self.dropout_backbone(x_left)

        x_right = self.backbone(x_right)
        x_right = self.dropout_backbone(x_right)
        # x.shape = (batch_size * num_frames, 1024)


        x_left = x_left.reshape(b, n, -1)
        x_right = x_right.reshape(b, n, -1)
        # x.shape = (batch_size, num_frames, 1024)
        _, (h_left, c_left) = self.neck(x_left)
        _, (h_right, c_right) = self.neck(x_right)
        x_left = self.dropout_neck(h_left[-1])
        x_right = self.dropout_neck(h_right[-1])
        y_left = self.head(x_left)  # h[-1].shape = (batch_size, 1024)
        y_right = self.head(x_right)  # h[-1].shape = (batch_size, 1024)
        return y_left, y_right


if __name__=="__main__":
    data_dir = Path("input/data_10hz")
    frame_label_df = pd.read_csv(data_dir / "frame_label.csv")
    with open(data_dir.parent / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1

    max_height, max_width = get_max_size(frame_label_df, data_dir / "frames")

    dataset = Dataset1(
        frame_dir=data_dir / "frames",
        frame_label_df=frame_label_df,
        time_window=2,
        width=4,
        max_height=max_height,
        max_width=max_width,
        action_to_id=metadata["action_to_id"],
        num_classes=num_classes,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model1(num_classes=num_classes, backbone="resnet34d", pretrained=False)
    model = model.to(device)
    for frames, label_left, label_right in loader:
        frames = frames.to(device).float()
        label_left = label_left.to(device)
        label_right = label_right.to(device)
        print(frames.shape)
        print(label_left.shape, label_right.shape)
        y_left, y_right = model(frames)
        print(y_left.shape, y_right.shape)

        break
