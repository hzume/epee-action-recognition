import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model0(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "resnet34d"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.neck = nn.LSTM(self.backbone.num_features, 1024, 1, batch_first=True)
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x, _ = self.neck(x)
        x = self.head(x)
        return x


class LitModel(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
