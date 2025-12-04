🌿 Smart Guava Leaf Disease Detection System
AI-powered early disease identification using MobileNetV2 & TFLite

a) Overview

The Smart Guava Leaf Disease Detection System is an intelligent deep-learning–based solution that detects six different types of guava leaf anomalies using a lightweight MobileNetV2 CNN model.
The model is optimized and converted into TensorFlow Lite (TFLite) format to run smoothly on low-power devices and web interfaces.
A modern Gradio UI provides easy image upload, prediction results, probability graphs, and preventive suggestions.
This system helps farmers, researchers, and students classify guava leaf diseases quickly and accurately.

b)  Detected Leaf Classes

The system can identify the following 6 classes:
Anthracnose, Healthy, Insect Bite, Multiple Infection, Scorch, Yellow Leaf Disease (YLD)

c) Key Features

1. MobileNetV2 trained model optimized for speed and accuracy
2. TFLite inference engine for lightweight deployment
3. Modern Gradio web interface
4. Probability bar graph showing prediction confidence
5. Preventive measures & causes displayed automatically

d) System Architecture

User Uploads Image → Preprocessing → TFLite Interpreter → CNN Prediction → Probability Graph Generation → Preventive Measures → Output to UI


e) Project Structure

Smart-Guava-Leaf-Disease-Detection/
│── app.py
│── model.tflite
│── style.css
│── requirements.txt
│── README.md
│── /sample_images
└── /screenshots


f) Technologies Used

Python
TensorFlow & TFLite
NumPy
Pillow
Gradio
Matplotlib

g) Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/Ishu-26/Guava-leaf-disease-detection-system.git
cd Smart-Guava-Leaf-Disease-Detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
python app.py

4️⃣ Access in Browser

Open:
http://127.0.0.1:7860


 i) Preventive Measures Example

Each disease class displays automatically generated advice, such as:

Anthracnose

Remove infected leaves
Use copper-based fungicides
Improve airflow around plant

YLD

Maintain soil nutrients
Treat with balanced fertilizers
Monitor for nutrient deficiencies

…and similar for all classes.

j) Future Scope

 1. Mobile App deployment (Android/iOS)
 2. Integration with IoT sensors & ESP32-CAM for real-time farm scanning
 3. Cloud deployment using AWS/GCP
 4. Farmer dashboard with disease analytics
 5. Auto-update model with new datasets
 6. Compatible with local deployment or cloud hosting
