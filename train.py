from pathlib import Path
from PIL import Image
from typing import Callable
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
import pytorch_lightning as pl
import src.hsv_transforms as T_hsv
from src.nnet import ResNetUNet


def main():
    transform = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(0.5)
    ])
    hsv_transform = T_hsv.HSVTransform([
        T_hsv.RandomCutOut(0.5, 0.2, 0.8, 0.25, 0.75, fill_value=0.5),
        T_hsv.WhiteNoise(0.2)
    ])
    dset = CarsDataset("data/cars196/cars_train", transform, hsv_transform)
    train_dset, val_dset = random_split(dset, [0.8, 0.2])
    train_dl = DataLoader(train_dset, 32, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_dset, 32, shuffle=False, num_workers=8)

    model = UNetModule(freeze=4)
    trainer = pl.Trainer(max_epochs=5, accelerator="gpu")
    trainer.fit(model, train_dl, val_dl)


class CarsDataset(Dataset):
    def __init__(self, root: Path | str, transform: Callable | None,
                 hsv_transform: T_hsv.HSVTransform, img_format: str = ".jpg"):
        self.root = Path(root)
        self.image_paths = sorted(
            [p for p in self.root.iterdir() if p.suffix == img_format],
            key=lambda p: p.stem
        )
        self.transform = transform
        self.hsv_transform = hsv_transform
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        img = np.array(img)
        rgb, hsv, target = self.hsv_transform(img)
        rgb = self.to_tensor(rgb)
        rgb = self.normalize(rgb)
        return rgb, hsv, target


class UNetModule(pl.LightningModule):
    def __init__(self, freeze: int = 5):
        super().__init__()
        self.model = ResNetUNet(freeze=freeze)
        self.loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        return self.model(x)

    def _forward_step(self, batch):
        inp, _, target = batch
        prediction = self(inp)
        return self.loss_fn(prediction, target)

    def training_step(self, batch, batch_idx):
        loss = self._forward_step(batch)
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._forward_step(batch)
        self.log("val_loss", loss.item())
        return loss


if __name__ == "__main__":
    main()
