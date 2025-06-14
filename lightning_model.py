import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import Accuracy
import lightning.pytorch as L


class ASL3DConvNet(nn.Module):
  def __init__(self, classes):
    super(ASL3DConvNet, self).__init__()
    self.conv1 = self._make_conv_layer(2, 8)
    self.conv2 = self._make_conv_layer(8, 16)
    self.conv3 = self._make_conv_layer(16, 32)

    self.adaptive_pool = nn.AdaptiveAvgPool3d((6, 8, 8))

    self.fc1 = nn.Linear(32 * 6 * 8 * 8, 2048)
    self.fc2 = nn.Linear(2048, classes)

  def _make_conv_layer(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.adaptive_pool(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return {
        "logits": x,
        "probs": F.softmax(x, dim=1)
    }
  
class SignLanguageRecognition(L.LightningModule):
  def __init__(self, lr, num_classes=2000):
    super().__init__()
    self.save_hyperparameters()
    self.model = ASL3DConvNet(num_classes)
    self.loss = nn.CrossEntropyLoss()
    self.acc = Accuracy(task="multiclass", num_classes=num_classes)

  def training_step(self, batch, batch_idx):
    events, targets = batch

    output = self.model(events)
    loss = self.loss(output["logits"], targets)
    self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    events, targets = batch

    output = self.model(events)
    loss = self.loss(output["logits"], targets)
    self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)

    accuracy = self.acc(output["probs"], targets)
    self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    return {"val_loss": loss, "val_acc": accuracy}

  def test_step(self, batch, batch_idx):
    events, targets = batch

    output = self.model(events)
    loss = self.loss(output["logits"], targets)
    accuracy = self.acc(output["probs"], targets)

    self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    return {"test_loss": loss, "test_acc": accuracy}

  def predict_step(self, batch, batch_idx):
    events, _ = batch
    output = self.model(events)
    return torch.argmax(output["probs"], dim=1)

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.hparams.lr)
  

def get_model():
  return SignLanguageRecognition.load_from_checkpoint("checkpoints/nwlasl-20-classes-version-4-time_window-normalized.ckpt")