import torch
from torch import nn, optim
from tqdm import tqdm
import os
import json

class Trainer:
    def __init__(self, model, train_loader, test_loader, device,
                 optimizer_cls=optim.Adam, lr=1e-3, epochs=10, loss_fn=None,
                 checkpoint_path="checkpoint.pth", log_path="training_log.json"):
        """
        کلاس آموزش با قابلیت ادامه از چک‌پوینت.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn if loss_fn else nn.BCEWithLogitsLoss()
        self.checkpoint_path = checkpoint_path
        self.log_path = log_path
        self.start_epoch = 1
        self.history = []

        # اگر چک‌پوینت قبلی هست، لود کن
        self._load_checkpoint()

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }
        torch.save(checkpoint, self.checkpoint_path)
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=4)

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            print(f"[INFO] Loading checkpoint from '{self.checkpoint_path}'...")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.history = checkpoint.get("history", [])
            self.start_epoch = checkpoint["epoch"] + 1
            print(f"[INFO] Resuming from epoch {self.start_epoch}")
        else:
            print("[INFO] No checkpoint found. Starting from scratch.")

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for x, y in loop:
            x, y = x.to(self.device), y.to(self.device).float()

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred.squeeze(), y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device).float()
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred.squeeze(), y)
                total_loss += loss.item()

                predicted = (torch.sigmoid(y_pred) > 0.5).int()
                correct += (predicted.squeeze() == y.int()).sum().item()
                total += y.size(0)

        accuracy = 100 * correct / total
        return total_loss / len(self.test_loader), accuracy

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.evaluate()

            # ذخیره در history
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            print(f"Epoch [{epoch}/{self.epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")

            # ذخیره چک‌پوینت
            self._save_checkpoint(epoch)
        return self.history
