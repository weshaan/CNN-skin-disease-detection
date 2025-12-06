# 🩺 Skin Cancer Detection Using DenseNet (Transfer Learning)

Early detection of skin cancer significantly improves treatment outcomes.  
This project leverages **DenseNet transfer learning** to classify dermoscopic images into **benign** and **malignant** categories.

---

## 📂 Project Overview

- **Goal:** Automatic skin cancer classification from dermoscopic images.  
- **Approach:** Transfer learning using **DenseNet** pretrained on ImageNet.  
- **Benefits:** High performance even with limited medical datasets.

### Key Features
- Automated preprocessing of HAM10000 dataset
- Transfer learning with DenseNet
- Fine-tuning for improved performance
- Model evaluation using:
  - Accuracy  
  - Precision, Recall, F1-score  
  - Confusion Matrix  
- Saved trained model ready for deployment

---

## 📌 Dataset: HAM10000

**HAM10000** (Human Against Machine with 10000 training images) is a widely used dermatology dataset:

- **Total images:** 10,015  
- **Type:** Dermoscopic images  
- **Original classes:** 7 skin lesion types  
- **Used in this project:** Grouped into **Benign vs Malignant**

**Dataset Source:**  
[HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---

## 🧠 Model Architecture

**Base:** DenseNet pretrained on ImageNet (frozen layers initially)  
**Custom Head:**
- `GlobalAveragePooling2D`
- `Dense(256, activation='relu')`
- `Dropout(0.3)`
- `Dense(2, activation='softmax')`

**Training Details:**

| Parameter        | Value                                   |
|-----------------|-----------------------------------------|
| Optimizer        | Adam                                     |
| Learning Rate    | 1e-4 (Phase 1), 1e-5 (Phase 2)         |
| Loss Function    | Binary Crossentropy                      |
| Batch Size       | 32                                       |
| Image Size       | 224 × 224                                |
| Epochs           | 30 (10 + 20)                             |

**Why DenseNet?**
- Dense connections improve feature propagation and reduce vanishing gradients  
- Excellent at extracting detailed features for medical images  
- Outperforms ResNet in fine-grained image classification tasks

---

## 🚀 Training Pipeline

<img width="2011" height="290" alt="Untitled diagram-2025-12-06-164528" src="https://github.com/user-attachments/assets/d0d738c8-89fe-44e4-b21a-12ed573c79e7" />

---

## 📊 Performance Metrics

### Classification Report

| Class      | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|--------|
| Benign    | 0.90      | 0.86   | 0.88     | 360    |
| Malignant | 0.84      | 0.88   | 0.86     | 300    |

- **Overall Accuracy:** 87%  
- **Macro & Weighted Avg:** Precision, Recall, F1-score = 0.87  
- **Interpretation:** High recall for malignant cases ensures critical detections are not missed.

### Fine-Tuning Impact

| Metric                | Before Fine-Tuning | After Fine-Tuning |
|----------------------|-----------------|----------------|
| Training Accuracy     | ~56%            | ~85%           |
| Validation Accuracy   | ~50%            | ~82%           |
| Test Accuracy         | -               | ~88%           |
| Validation Loss       | 0.74            | 0.57           |
| AUC                   | 0.78            | 0.91           |

> Fine-tuning DenseNet significantly improves performance across all metrics, especially in critical test set evaluation.

---

## 🖼️ Confusion Matrix 

Visualize model predictions:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

![WhatsApp Image 2025-12-06 at 19 50 06_2a023192](https://github.com/user-attachments/assets/aaf55514-98a7-4813-8f5a-b479a256f94d)

## 🔧 Installation & Setup

### 1️⃣ Clone Repository
```
git clone https://github.com/your-username/skin-cancer-detection.git  
cd skin-cancer-detection
```
### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 3️⃣ Prepare Dataset
```
Place **HAM10000** inside the `/data/` directory  
OR run:  
python src/dataset_download.py
```
### 4️⃣ Train Model
```
python src/train.py
```
### 5️⃣ Evaluate Model
```
python src/evaluate.py
```
---

### 📦 Saved Model
```
**Path:** models/densenet_skin_cancer.h5
```

## Load for inference:
```python
import tensorflow as tf

model = tf.keras.models.load_model("models/densenet_skin_cancer.h5")
```
NOTE: over github, the model may be in compressed file. extract with a simple python script before loading.

---
## ⭐ Summary

This project demonstrates **automatic skin cancer detection** using **DenseNet-based transfer learning**.  
Through fine-tuning and feature extraction, the model achieves high accuracy and strong performance in identifying malignant skin lesions, making it well-suited for medical imaging and computer-aided diagnosis applications.
