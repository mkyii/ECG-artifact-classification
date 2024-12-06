import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from data_module import SignalDataModule
from model import CNN1D , ResNet1D, CNN1D_Modified
from cpi_metric import CPIMetric  # 수정된 CPI 메트릭 임포트
import mlflow.pytorch
import numpy as np
import mlflow
# 데이터 로드
import pickle

with open('../datasets/K-MEDICON_2024_SUB1_TRAINING_SET/Signal_Train.pkl', 'rb') as f:
    signal_train = pickle.load(f)
signal_train = np.array(signal_train)

with open('../datasets/K-MEDICON_2024_SUB1_TRAINING_SET/Target_Train.pkl', 'rb') as f:
    target_train = pickle.load(f)
target_train = np.array(target_train).flatten()

# 데이터 모듈 초기화
batch_size = 32
data_module = SignalDataModule(signal_train, target_train, batch_size=batch_size)

# 모델 초기화
# model = CNN1D()
# model = ResNet1D()
model = CNN1D_Modified()

# MLflow 설정
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("CNN1D-Signal-Train")

# MLFlowLogger 사용
mlf_logger = MLFlowLogger(experiment_name="CNN1D-Signal-Train")

# CPI 메트릭 설정 (수정된 CPIMetric 사용)
cpi_metric = CPIMetric()

# Trainer 설정
trainer = pl.Trainer(
    max_epochs=500,
    accelerator='gpu',
    logger=mlf_logger,
    callbacks=[cpi_metric]  # CPI 메트릭 추가
)

# 학습 시작
best_cpi = -np.inf  # 가장 높은 CPI 값을 추적

with mlflow.start_run() as run:
    for epoch in range(500):  # 에포크 수만큼 반복
        trainer.fit(model, datamodule=data_module)
        
        # 현재 에포크에서의 CPI 값을 로깅된 결과로부터 추출
        cpi_value = trainer.callback_metrics.get("CPI")  # CPI 값을 가져옴

        if cpi_value is not None and cpi_value > best_cpi:
            best_cpi = cpi_value
            mlflow.pytorch.log_model(model, f"models/best_model_epoch_{epoch}")
            print(f"Epoch {epoch}: Best model saved with CPI: {best_cpi}")

    # 최종 모델 저장 (테스트 후)
    trainer.test(model, datamodule=data_module)
    mlflow.pytorch.log_model(model, "models/final_model")
