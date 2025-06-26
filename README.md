# Disaster_Resilience

# 🌍 Disaster Resilience and Image Segmentation System

This project uses deep learning to classify disaster types (flood, fire, landslide) from aerial imagery and to segment impacted regions (e.g., water, roads, buildings) for resilience analysis using the FloodNet and other datasets.

---


## 📌 Objectives

- 🔍 **Classify** the type of disaster (Flood, Fire, Landslide) from aerial images.
- 🧠 **Segment** key features from images using a U-Net model (e.g., flooded area, buildings, vegetation, roads).
- 📊 **Count and report** segmented regions for situational awareness.

---

## 🧠 Model Summary

### 1. **Disaster Classifier**
- CNN-based 3-class classifier using TensorFlow/Keras.
- Classes: `Flood`, `Fire`, `Landslide`
- Input: 256x256 RGB image  
- Output: Disaster class

### 2. **Semantic Segmentation (U-Net)**
- U-Net with skip connections and softmax output.
- Classes: `Background`, `Buildings`, `Vegetation`, `Water`, `Roads`, `Vehicles`, `Flooded Area`
- Trained on mask-image pairs from FloodNet & Landslide datasets.

---

## 🧰 Tech Stack

- **Python** (TensorFlow, NumPy, PIL, scikit-learn)
- **Jupyter Notebook / Kaggle Kernel**
- **ONNX** for model export
- **Matplotlib** & **Seaborn** for visualization

---

## 📁 Dataset Sources

- [FloodNet Challenge Dataset (Track 1)](https://competitions.codalab.org/competitions/25293)
- [Fire Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
- [Landslide Dataset](https://www.kaggle.com/datasets)

Each dataset includes:
- Aerial RGB images (`.jpg`, `.png`)
- Segmentation masks with multi-class labels for floods/landslides

---

## 🗂 Project Structure

