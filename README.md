# 🚀 Signal Classification with PyTorch Lightning

Welcome to the **Signal Classification** project! This repository showcases powerful signal processing models built with **PyTorch Lightning**. Dive into the world of 1D Convolutional Neural Networks, Residual Networks, and Attention-Enhanced CNNs to tackle complex time-series data challenges.

---

## 📚 Table of Contents
- [✨ Overview](#-overview)
- [🛠️ Features](#-features)
- [🏗️ Architecture](#-architecture)
- [💻 Installation](#-installation)
- [🚀 Usage](#-usage)
- [📊 Model Training](#-model-training)
- [🧪 Evaluation](#-evaluation)
- [⚙️ Configuration](#-configuration)
- [🎓 K-MEDICON 2024 Participation](#-k-medicon-2024-participation)
- [📜 License](#-license)

---

## ✨ Overview
This project brings together cutting-edge AI techniques to classify signal data efficiently. From **deep CNNs** to **Residual Networks with attention mechanisms**, each model is optimized for high performance and adaptability.

---

## 🛠️ Features
- ✅ **SignalDataModule:** Simplifies dataset preparation and DataLoader management.
- ✅ **CNN1D:** Classic convolutional model for time-series signals.
- ✅ **ResNet1D:** Advanced ResNet architecture for deep feature extraction.
- ✅ **CNN1D_Modified:** Attention-powered CNN for improved global feature learning.
- ✅ **Stratified Data Splitting:** Ensures balanced train/validation/test datasets.

---

## 🏗️ Architecture
### 📊 **CNN1D:**
- 🔹 Two convolutional layers with pooling.
- 🔹 Dense fully connected layers for classification.

### 🧠 **ResNet1D:**
- 🔹 Residual blocks for in-depth feature extraction.
- 🔹 Adaptive average pooling.

### 🌟 **CNN1D_Modified:**
- 🔹 Multi-layered convolution and pooling.
- 🔹 Attention mechanism for refined global feature learning.
- 🔹 Fully connected output layers.

---

## 💻 Installation
Get started in a snap! 🚀
```bash
$ git clone https://github.com/your-username/signal-classification.git
$ cd signal-classification
$ pip install -r requirements.txt
```

---

## 🚀 Usage
### 📊 Data Preparation
Ensure your dataset follows this structure:
- `signal_train`: Numpy array `(samples, sequence_length, channels)`.
- `target_train`: Corresponding labels.

### 🏋️ Model Training Example
```python
from SignalDataModule import SignalDataModule
from CNN1D import CNN1D

# Initialize DataModule
signal_data = SignalDataModule(signal_train, target_train)

# Initialize Model
model = CNN1D()

# Train the model
from pytorch_lightning import Trainer
trainer = Trainer(max_epochs=50)
trainer.fit(model, signal_data)
```

---

## 📊 Model Training
Start training your model with ease:
```bash
$ python train.py --model CNN1D --epochs 50
```

---

## 🧪 Evaluation
Evaluate the performance of your trained model:
```bash
$ python evaluate.py --model CNN1D --checkpoint path/to/checkpoint.ckpt
```

---

## ⚙️ Configuration
Fine-tune hyperparameters in `config.yaml`:
```yaml
batch_size: 32
learning_rate: 0.001
max_epochs: 50
```

---

## 🎓 K-MEDICON 2024 Participation
We are proud to announce our participation in **K-MEDICON 2024**, hosted by **Korea University Medical Big Data Research Institute**.

### 📅 **Event Details:**
- **Registration:** August 13, 2024 – August 19, 2024
- **Final Participant Selection:** August 22, 2024
- **Data Release & Competition:** August 28, 2024 – October 23, 2024
- **Results Announcement & Awards:** November 2024

### 🧠 **Competition Topics:**
1️⃣ **ECG Signal Classification:** Artifact-included 12-lead ECG signal classification.
2️⃣ **Pathology Image Analysis:** Giga-pixel whole slide image analysis in bladder tumors.

This event brings together the brightest minds in digital healthcare, AI, and medical data research. Join us in pushing the boundaries of AI-driven healthcare solutions!

---

## 📜 License
This project is licensed under the **MIT License**.

---

### 🤝 Contributing
Found a bug or have a feature request? Open an issue or submit a pull request!

**Author:** [Your Name]  
**Email:** [Your Email]

🌟 *Star this repository if you found it useful!* 🚀
