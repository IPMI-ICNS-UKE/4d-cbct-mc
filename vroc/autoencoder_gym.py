import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from vroc.models import AutoEncoder


class AutoencoderGym:
    def __init__(self, train_loader, val_loader, device, out_path):
        self.device = device
        self.out_path = out_path
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = AutoEncoder().to(device=self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = torch.cuda.amp.GradScaler()
        self.save_best_model = SaveBestModel(out_path=out_path)
        self.writer = SummaryWriter()

    def workout(self, n_epochs=100, validation_epoch=5):
        pbar = trange(1, n_epochs + 1, unit="epoch")
        val_loss = np.NAN
        epoch_loss = np.NAN
        val_losses = []
        epoch_losses = []

        for epoch in pbar:
            pbar.set_description(f"Epoch: {epoch}")
            pbar.set_postfix_str(f"loss: {epoch_loss:.5f}, val. loss: {val_loss:.5f}")
            running_loss = self._train()
            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_losses.append(epoch_loss)

            if epoch % validation_epoch == 0:
                val_loss = self._validation()
                self.save_best_model(val_loss, epoch, self.model, self.optimizer)
                val_losses.append(val_loss)

            self.writer.add_scalar("Loss/train", epoch_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)

    def _validation(self):
        val_loss = 0.0
        for data, _ in self.val_loader:
            self.model.eval()
            with torch.no_grad():
                images = data.to(self.device)
                outputs, embedded = self.model(images)
                outputs = torch.sigmoid(outputs)
                loss = self.criterion(outputs, images)
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(self.val_loader.dataset)
        return val_loss

    def _save_model(self, epoch, val_loss):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(
            state,
            os.path.join(
                self.out_path, f"epoch{epoch:03d}_val_loss_=_{val_loss:.03f}.pth"
            ),
        )

    def _train(self):
        running_loss = 0.0
        self.model.train()
        for data, _ in self.train_loader:
            images = data.to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs, _ = self.model(images)
                outputs = torch.sigmoid(outputs)
                loss = self.criterion(outputs, images)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item() * images.size(0)
        return running_loss


class SaveBestModel:
    """Class to save the best model while training.

    If the current epoch's validation loss is less than the previous
    least less, then save the model state.
    """

    def __init__(self, out_path, best_valid_loss=float("inf")):
        self.best_valid_loss = best_valid_loss
        self.out_path = out_path

    def __call__(self, current_valid_loss, epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(self.out_path, "best_model.pth"),
            )
