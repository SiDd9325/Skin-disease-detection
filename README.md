# 🩺 Skin Cancer Classification using EfficientNet (HAM10000)

This project provides a **step-by-step guide** to load a pre-trained skin cancer classification model (`skin_cancer_model_rgb_v2.h5`) and run predictions on single or multiple images from the **HAM10000 dataset** (or your own images).

The model classifies lesions into **7 classes**:
- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)

> ⚠️ **Disclaimer:** This model is for **educational and research purposes only**. It is **not a medical diagnostic tool**.

---

## 📦 1. Dependencies & Installation

### 1.1 Recommended Setup (Conda Environment)

Create and activate a new Python environment to avoid dependency conflicts:

```bash
conda create -n skinenv python=3.10 -y
conda activate skinenv
````

### 1.2 Install Required Libraries

Install the necessary packages:

```bash
pip install --upgrade pip
pip install tensorflow pillow matplotlib pandas tqdm
```

### 1.3 Verify Installation

Run:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

Make sure TensorFlow loads without errors (should be ≥ 2.14 for Keras 3 compatibility).

---

## 📂 2. Project Structure

Your folder should look like this:

```
project/
 ├─ skin_cancer_model_rgb_v2.h5       # Pre-trained model file
 ├─ predict.py                        # Main prediction script
 ├─ README.md                         # This file
 └─ HAM10000_images_part_2/           # Folder containing test images
```

---

## ▶️ 3. Running Predictions

### 3.1 Predicting a Single Image

Run:

```bash
python predict.py --image "HAM10000_images_part_2/ISIC_0034264.jpg"
```

Example Output:

```
Loading model from skin_cancer_model_rgb_v2.h5 ...
Model loaded successfully.

Top 3 predictions for: HAM10000_images_part_2/ISIC_0034264.jpg
  Melanocytic nevi (nv): 85.23%
  Benign keratosis-like lesions (bkl): 10.14%
  Melanoma (mel): 4.63%
```

---

### 3.2 Predicting an Entire Folder

To classify all images in a folder and save results to a CSV file:

```bash
python predict.py --folder "HAM10000_images_part_2" --csv results.csv
```

This will:

* Process all `.jpg`, `.jpeg`, `.png` files inside the folder.
* Print top-3 predictions for each image.
* Save results to `results.csv` in the current folder.

Sample `results.csv` output:

| filename         | top1_name             | top1_prob | top2_name                           | top2_prob | top3_name                           | top3_prob |
| ---------------- | --------------------- | --------- | ----------------------------------- | --------- | ----------------------------------- | --------- |
| ISIC_0034264.jpg | Melanocytic nevi (nv) | 0.8523    | Benign keratosis-like lesions (bkl) | 0.1014    | Melanoma (mel)                      | 0.0463    |
| ISIC_0025789.jpg | Melanoma (mel)        | 0.7021    | Melanocytic nevi (nv)               | 0.2123    | Benign keratosis-like lesions (bkl) | 0.0856    |

---

## 🏷️ 4. Class Labels

| Index | Full Name                                                         | Short   |
| ----: | ----------------------------------------------------------------- | ------- |
|     0 | Actinic keratoses and intraepithelial carcinoma / Bowen's disease | `akiec` |
|     1 | Basal cell carcinoma                                              | `bcc`   |
|     2 | Benign keratosis-like lesions                                     | `bkl`   |
|     3 | Dermatofibroma                                                    | `df`    |
|     4 | Melanoma                                                          | `mel`   |
|     5 | Melanocytic nevi                                                  | `nv`    |
|     6 | Vascular lesions                                                  | `vasc`  |

---

## 📜 5. Training Summary (Optional)

The model was trained using **EfficientNetV2B0** with:

* **Image size:** 224x224 RGB
* **Optimizer:** Adam
* **Loss:** Categorical Crossentropy
* **Callbacks:** EarlyStopping, ModelCheckpoint
* **Augmentation:** Rotation, shift, zoom, horizontal flip

Saved as a `.h5` file for easy loading in Keras 3.

---

## 🧠 6. Notes

* Predictions are probabilities from the softmax output.
* If you get **low accuracy**, retrain the model or fine-tune with more data.
* Ensure your input image size matches training size (224×224).
* This code works on CPU and GPU (if TensorFlow GPU is installed).

---

## 🛠️ 7. Troubleshooting

| Problem                                   | Solution                                                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `OSError: SavedModel file does not exist` | Make sure you use `tf.keras.models.load_model("skin_cancer_model_rgb_v2.h5")` with `.h5` models (not `TFSMLayer`). |
| `ValueError: File format not supported`   | Use `.h5` or `.keras` models with Keras 3. Re-save your model if needed.                                           |
| SSE/AVX warnings                          | These are CPU optimization hints. Ignore unless you want to build TensorFlow from source.                          |
| Slow predictions                          | Install TensorFlow GPU if you have an NVIDIA GPU + CUDA support.                                                   |

---

## 🧾 8. Example Execution Flow

1. **Clone/Download the project.**
2. **Create a virtual environment & install dependencies.**
3. Place `skin_cancer_model_rgb_v2.h5` in the project folder.
4. Put test images inside a folder (e.g., `HAM10000_images_part_2`).
5. Run predictions for a single image or the entire folder.
6. (Optional) Export results to a CSV for further analysis.

---

## 👨‍💻 Author

Maintained by **Siddarth (Engineer)** — feel free to fork and modify this project for research and learning purposes.

```

---

