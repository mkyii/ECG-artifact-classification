# cpi_metric.py

import torch
from sklearn.metrics import matthews_corrcoef
from torchmetrics.classification import BinaryF1Score
from torchmetrics.functional import auroc
import pytorch_lightning as pl

class CPIMetric(pl.Callback):
    def __init__(self):
        super().__init__()
        self.probs = []
        self.targets = []
        self.f1_metric = BinaryF1Score()  # BinaryF1Score 정의

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 'outputs'에서 logits와 labels를 안전하게 가져오기
        logits = outputs['logits'] if 'logits' in outputs else None
        labels = outputs['labels'] if 'labels' in outputs else None
        
        if logits is not None and labels is not None:
            probs = torch.sigmoid(logits).squeeze()  # 이진 분류를 위한 sigmoid 적용
            self.probs.append(probs)
            self.targets.append(labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        if len(self.probs) > 0:
            device = pl_module.device  # 모델이 있는 디바이스 확인

            probs = torch.cat(self.probs).to(device)  # GPU로 이동
            targets = torch.cat(self.targets).to(device)  # GPU로 이동

            # F1 Score 계산 (BinaryF1Score 사용)
            self.f1_metric = self.f1_metric.to(device)
            f1_result = self.f1_metric(probs, targets)

            # AUROC 계산
            auroc_result = auroc(probs, targets, task='binary')

            # MCC (Matthews Correlation Coefficient) 계산
            preds = (probs > 0.5).cpu().numpy()  # 이진화된 값
            targets_np = targets.cpu().numpy()
            mcc_result = torch.tensor(matthews_corrcoef(targets_np, preds)).to(device)

            # CPI = 0.25 * F1 Score + 0.25 * AUROC + 0.5 * MCC
            cpi_result = 0.25 * f1_result + 0.25 * auroc_result + 0.5 * mcc_result

            # Log metrics
            pl_module.log('F1 Score', f1_result)
            pl_module.log('AUROC', auroc_result)
            pl_module.log('MCC', mcc_result)
            pl_module.log('CPI', cpi_result)

        # Reset for next epoch
        self.probs = []
        self.targets = []
