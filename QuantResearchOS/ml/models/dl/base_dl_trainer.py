import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class BaseDLTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float = 1e-3
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()  # Assuming regression for baseline

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # --- Validation ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            logger.info(
                "Epoch %d/%d, Train Loss: %.4f, Val Loss: %.4f",
                epoch + 1,
                epochs,
                avg_train_loss,
                avg_val_loss,
            )

    def predict(self, test_loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.append(outputs.cpu())
        return torch.cat(predictions, dim=0)
