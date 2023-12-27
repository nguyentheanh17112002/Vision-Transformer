from typing import Any, Dict, List, Tuple
from lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader


class CifarDataModule(LightningDataModule):
    def __init__(self, data_dir:str = './data', batch_size:int = 64) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.num_classes = 10

    def prepare_data(self) -> None:
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)
        return super().prepare_data()
    
    def setup(self, stage: str = None) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
        return super().setup(stage)
    
    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
    
if __name__ == "__main__":
    _ = CifarDataModule("/home/anhnt596/VisionTransformer/data", 64)