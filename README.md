# 🩺 AI Skin Disease Prediction System (HAM10000)

This project is an **AI-based skin cancer classification system** built using **TensorFlow/Keras** and trained on the **HAM10000 dataset**.
It classifies dermatoscopic images into **7 skin disease categories** using a fine-tuned **EfficientNetB0** model.

---

## 📂 Project Structure

```
.
├── dataset/                # HAM10000 images (download & place here)
│   ├── train/
│   ├── val/
│   └── test/
├── model.py                # Training script
├── predict.py              # Prediction script
├── skin_cancer_model.h5    # Saved trained model
└── README.md               # This file
```

---

## 🛠️ Installation

Make sure you have **Python 3.9+** installed.
Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

### ✅ Required Dependencies

Your `requirements.txt` should include:

```
tensorflow>=2.15
keras>=3.0
numpy>=1.26
pandas>=2.0
matplotlib>=3.8
scikit-learn>=1.3
opencv-python>=4.9
Pillow>=10.0
h5py>=3.10
```

These cover:

* **TensorFlow/Keras** – model training, EfficientNetB0, saving/loading
* **NumPy/Pandas** – data manipulation
* **Matplotlib** – visualizing training performance
* **scikit-learn** – metrics, confusion matrix
* **OpenCV** – image loading and resizing
* **Pillow** – image preprocessing
* **h5py** – for `.h5` model saving/loading

---

## 📥 Dataset Setup

1. Download **HAM10000 dataset** from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Organize into the following folder structure:

```
dataset/
├── train/
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
├── val/
│   └── (same subfolders as train)
└── test/
    └── (same subfolders as train)
```

Recommended split:

* **70% train**, **20% validation**, **10% test**

---

## 🏋️ Training the Model

Run the training script:

```bash
python model.py
```

This will:

* Load and preprocess RGB images (224×224)
* Train **EfficientNetB0** (ImageNet-pretrained)
* Freeze base layers, fine-tune classifier
* Save the trained model as `skin_cancer_model.h5`

---

## 🔮 Running Predictions

Once the model is trained, run predictions with:

```bash
python predict.py --image path_to_image.jpg
```

This will:

* Load the saved model
* Preprocess the image
* Output predicted class and confidence

---

## 📊 Expected Results

* **Accuracy:** ~80–85% (depends on epochs and dataset split)
* **Input Size:** `224x224x3` (RGB)
* **Output Classes:** 7 (akiec, bcc, bkl, df, mel, nv, vasc)

---

## ⚠️ Notes

* Always use **RGB mode** (`color_mode='rgb'`) to avoid weight shape mismatch errors.
* Use `.h5` or `.keras` format for saving models with Keras 3.
* If using a GPU, ensure **CUDA/cuDNN** are installed for faster training.

---

Would you like me to generate a **ready-to-use `requirements.txt` file** with these exact dependencies (with versions pinned for compatibility) so you can just install it directly?
