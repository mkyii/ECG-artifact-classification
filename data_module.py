import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

class SignalDataModule(pl.LightningDataModule):
    def __init__(self, signal_train, target_train, batch_size=32):
        super().__init__()
        self.signal_train = np.array(signal_train)  # numpy로 변환
        self.target_train = np.array(target_train).flatten()  # numpy로 변환 후 flatten
        self.batch_size = batch_size
        
        # Train, validation, test datasets 초기화
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # 데이터셋을 stratify하여 train/val/test로 나누기 (8:1:1 비율)
        if stage == 'fit' or stage is None:
            X_train, X_temp, y_train, y_temp = train_test_split(
                self.signal_train, self.target_train, test_size=0.2, stratify=self.target_train, random_state=42
            )
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
            )
            
            # TensorDataset 생성
            self.train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
            self.val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
            self.test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
