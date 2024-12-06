import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CNN1D(pl.LightningModule):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=16, stride=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, stride=1, padding=0)
        self.pool = nn.MaxPool1d(kernel_size=2)  # Pooling size 2 will reduce the sequence length by half

        # 수정된 fc1_input_size: Conv 레이어와 MaxPool 레이어의 출력 크기 계산
        conv1_output_size = (5000 - 16 + 1) // 1
        pool1_output_size = conv1_output_size // 2
        conv2_output_size = (pool1_output_size - 16 + 1) // 1
        pool2_output_size = conv2_output_size // 2

        self.fc1_input_size = 128 * pool2_output_size
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 1)  # 이진 분류이므로 출력 노드를 1로 수정

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)            # 마지막 레이어에서는 sigmoid를 적용하지 않음
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()  # 출력이 (batch_size, 1)이므로 squeeze()로 차원 축소
        loss = F.binary_cross_entropy_with_logits(logits, y.float())  # BCEWithLogitsLoss 사용
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log('val_loss', loss)
        return {'logits': logits, 'labels': y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        self.log('test_loss', loss)
        return {'logits': logits, 'labels': y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=61, stride=1, padding=30):  # 패딩을 8로 수정하여 출력 크기 조정
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 입력과 출력의 크기가 일치해야 함
        out = self.relu(out)

        return out

class ResNet1D(pl.LightningModule):
    def __init__(self):
        super(ResNet1D, self).__init__()
        self.layer1 = BasicBlock(12, 64, kernel_size=61, stride=1, padding=30)  # 패딩을 8로 수정하여 출력 크기 맞춤
        self.layer2 = BasicBlock(64, 128, kernel_size=61, stride=1, padding=30)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 마지막 풀링 레이어
        self.fc = nn.Linear(128, 1)  # 이진 분류를 위한 출력 레이어

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Conv1d에 맞게 (batch_size, sequence_length, in_channels) -> (batch_size, in_channels, sequence_length)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = nn.BCEWithLogitsLoss()(logits, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = nn.BCEWithLogitsLoss()(logits, y.float())
        self.log('val_loss', loss)
        return {'logits': logits, 'labels': y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        loss = nn.BCEWithLogitsLoss()(logits, y.float())
        self.log('test_loss', loss)
        return {'logits': logits, 'labels': y}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class CNN1D_Modified(pl.LightningModule):
    def __init__(self, input_channels=12, seq_length=5000):  # seq_length 추가
        super(CNN1D_Modified, self).__init__()

        # Local Feature Learning Part
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)

        # Reduce sequence length after pooling
        self.seq_length = seq_length
        self.flatten_size = self.calculate_flatten_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, 448)
        self.fc3 = nn.Linear(448, 2)

        # Reduce the attention layer size
        reduced_attention_size = 512  # Reduce this based on your memory requirements
        self.attention_layer = nn.Linear(self.flatten_size, reduced_attention_size)

    def calculate_flatten_size(self):
        # Calculate the flatten size after two max poolings
        pooled_seq_length = self.seq_length // (2 * 2)
        return 128 * pooled_seq_length

    def forward(self, x):
        # Local Feature Learning Part
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Global Feature Learning with Attention
        attention_weights = torch.softmax(self.attention_layer(x), dim=1)
        attention_applied = attention_weights * x

        # Fully connected layers
        x = F.relu(self.fc1(attention_applied))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        device = x.device  # Ensure batch is on the same device as model
        logits = self(x.to(device))
        loss = F.cross_entropy(logits, y.to(device))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device  # Ensure batch is on the same device as model
        logits = self(x.to(device))
        loss = F.cross_entropy(logits, y.to(device))
        self.log('val_loss', loss)
        return {'logits': logits, 'labels': y}

    def test_step(self, batch, batch_idx):
        x, y = batch
        device = x.device  # Ensure batch is on the same device as model
        logits = self(x.to(device))
        loss = F.cross_entropy(logits, y.to(device))
        self.log('test_loss', loss)
        return {'logits': logits, 'labels': y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
