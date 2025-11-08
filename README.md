# 🦷 Dental Caries Detection using CNN

## 📘 Overview
Dental caries (commonly known as tooth decay) remains one of the most prevalent oral health problems worldwide.  
This project presents an **AI/ML-based web application** that uses **Convolutional Neural Networks (CNNs)** to automatically detect dental caries from **intraoral X-ray and dental images**.  

The goal is to assist dental professionals by providing an intelligent screening system that improves early diagnosis and reduces manual error.

---

## 🎯 Objectives
- Develop a **CNN model** capable of identifying dental caries from X-ray and intraoral images.  
- Preprocess and enhance medical images using **OpenCV**.  
- Deploy the trained model as a **web application** for real-time caries detection.  
- Evaluate model accuracy and reliability for academic and clinical research.

---

## 🧠 Methodology

### 1. Data Collection & Preprocessing
- **Dataset:** Collection of dental X-ray or intraoral images (labeled as *carious* / *non-carious*).  
- **Image preprocessing:**
  - Grayscale conversion  
  - Noise reduction (Gaussian blur)  
  - Contrast enhancement  
  - Image resizing and normalization using OpenCV  

### 2. Model Development
- **Framework:** TensorFlow / Keras  
- **Architecture:** Custom CNN model or transfer learning (e.g., VGG16 / ResNet50)  
- **Layers:**
  - Convolutional + ReLU activation  
  - MaxPooling  
  - Dropout for regularization  
  - Fully connected layers with Softmax activation  
- **Loss Function:** `categorical_crossentropy`  
- **Optimizer:** `Adam`  
- **Metrics:** Accuracy, Precision, Recall, F1-Score  

### 3. Model Training
- Training and validation split (e.g., 80:20)  
- Early stopping and learning rate scheduling  
- Data augmentation to improve generalization  

### 4. Deployment
- **Web Framework:** Flask / Streamlit  
- The web app allows users to upload a dental image, runs the trained CNN model, and displays:  
  - Prediction result (Caries / No Caries)  
  - Confidence score  
  - Visualization of detected regions (if implemented with Grad-CAM or heatmaps)

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3.10 |
| **Frameworks** | TensorFlow, Keras |
| **Image Processing** | OpenCV |
| **Web App** | Flask |
| **Visualization** | Matplotlib, Seaborn |
| **Data Handling** | NumPy, Pandas |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vijaykakde/Dental_Caries.git
cd dental-caries-detection 
pip install -r requirements.txt
python app.py

