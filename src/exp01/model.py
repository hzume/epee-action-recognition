import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import json
from torch.utils.data import DataLoader

from dataset import Dataset0, get_max_size

class Model0(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet34d", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.neck = nn.LSTM(self.backbone.num_features, 1024, 1, batch_first=True)
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, num_frames, height, width, channels)
        b, n, h, w, c = x.shape
        x = x.reshape(b * n, h, w, c)
        x = x.permute(0, 3, 1, 2)
        x = self.backbone(x)
        # x.shape = (batch_size * num_frames, 1024)
        x = x.reshape(b, n, -1)
        # x.shape = (batch_size, num_frames, 1024)
        _, (h, c) = self.neck(x)
        y = self.head(h[-1])  # h[-1].shape = (batch_size, 1024)
        return y


if __name__=="__main__":
    data_dir = Path("input/data")
    frame_label_df = pd.read_csv(data_dir / "frame_label.csv")
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1

    max_height, max_width = get_max_size(frame_label_df, data_dir / "frames")

    dataset = Dataset0(
        frame_dir=data_dir / "frames",
        frame_label_df=frame_label_df,
        time_window=2,
        max_height=max_height,
        max_width=max_width,
        action_to_id=metadata["action_to_id"],
        num_classes=num_classes,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model0(num_classes=num_classes, backbone="resnet34d", pretrained=False)
    model = model.to(device)
    for frames, label in loader:
        frames = frames.to(device).float()
        label = label.to(device)
        print(frames.shape)
        print(model(frames).shape, label.shape)

        break
