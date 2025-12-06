# Skin Cancer Detection (CNN)

**Description**
A simple, modular project that trains a CNN to classify skin lesion images (benign vs malignant).
This repository contains code for preprocessing, training, evaluation, and a small Flask app for inference.

**Structure**
See the repository layout in the project root.

**Quick start (local)**
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate      # on Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

2. Prepare data:
Place raw images into `data/raw/` organized as:
```
data/raw/class_name/image1.jpg
data/raw/class_name/image2.jpg
```
Then run preprocessing:
```bash
python scripts/run_preprocessing.py --src data/raw --dst data/processed --img-size 224 --val-split 0.2 --test-split 0.1
```

3. Train:
```bash
python scripts/run_training.py --data_dir data/processed --epochs 10 --batch_size 32 --save models/saved_model.h5
```

4. Run inference server:
```bash
python deployment/flask_app/app.py
# then open http://127.0.0.1:5000 and upload an image
```

**Notes**
- The code relies on Keras `ImageDataGenerator` for simplicity.
- For real medical projects, obtain proper datasets (ISIC / Kaggle) and follow ethical guidelines.
