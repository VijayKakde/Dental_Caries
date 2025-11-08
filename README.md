🦷 Dental Caries Detection using CNN
📘 Overview
Dental caries (commonly known as tooth decay) remains one of the most prevalent oral health problems worldwide.
This project presents an AI/ML-based web application that uses Convolutional Neural Networks (CNNs) to automatically detect dental caries from intraoral X-ray and dental images.
The goal is to assist dental professionals by providing an intelligent screening system that improves early diagnosis and reduces manual error.

🎯 Objectives
Develop a CNN model capable of identifying dental caries from X-ray and intraoral images.
Preprocess and enhance medical images using OpenCV.
Deploy the trained model as a web application for real-time caries detection.
Evaluate model accuracy and reliability for academic and clinical research.

🧠 Methodology
1. Data Collection & Preprocessing
Dataset: Collection of dental X-ray or intraoral images (labeled as carious / non-carious).
Image preprocessing:
Grayscale conversion
Noise reduction (Gaussian blur)
Contrast enhancement
Image resizing and normalization using OpenCV

2. Model Development
Framework: TensorFlow / Keras
Architecture: Custom CNN model or transfer learning (e.g., VGG16 / ResNet50)
Layers:
Convolutional + ReLU activation
MaxPooling
Dropout for regularization
Fully connected layers with Softmax activation
Loss Function: categorical_crossentropy
Optimizer: Adam
Metrics: Accuracy, Precision, Recall, F1-Score

4. Model Training
Training and validation split (e.g., 80:20)
Early stopping and learning rate scheduling
Data augmentation to improve generalization

4. Deployment
Web framework: Flask / Streamlit
The web app allows users to upload a dental image, runs the trained CNN model, and displays:
Prediction result (Caries / No Caries)
Confidence score
Visualization of detected regions (if implemented with Grad-CAM or heatmaps)

🧩 Tech Stack
Component	Technology
Language	Python 3.10
Frameworks	TensorFlow, Keras
Image Processing	OpenCV
Web App	Flask 
Visualization	Matplotlib, Seaborn
Data Handling	NumPy, Pandas

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/<your-username>/dental-caries-detection.git
cd dental-caries-detection

2. Install Dependencies
pip install -r requirements.txt

3. Run the Web Application
python app.py


Then open http://localhost:5000
 in your browser.

📊 Results
Metric	Value (Example)
Accuracy	94.6%
Precision	92.1%
Recall	90.7%
F1-Score	91.4%

Visualizations like training curves and confusion matrix can be added here.

🧪 Future Improvements
Incorporate segmentation to localize caries regions.
Use larger, more diverse datasets for better generalization.
Integrate Grad-CAM visual explanations for transparency.
Deploy on cloud (e.g., AWS / GCP / Hugging Face Spaces).

📚 References
S. Acharya et al., “Deep Learning for Caries Detection in Dental X-rays”, IEEE Access, 2021.
Krizhevsky et al., “ImageNet Classification with Deep Convolutional Neural Networks”, NIPS, 2012.
TensorFlow and OpenCV official documentation.

👨‍💻 Author

Vijay Kakde
AI/ML Enthusiast | Dental Imaging Research
📧 vijaykumarkakde77@gmail.com
