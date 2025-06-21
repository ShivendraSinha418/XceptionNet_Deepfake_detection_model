# 🎭 Deepfake Detection Using XceptionNet

A deep learning project that detects **deepfake videos** by leveraging the power of **XceptionNet** — fine-tuned on a custom dataset — and wrapped in an interactive **Flask interface** for easy inference.

---

![Deepfake Detection Banner](https://github.com/ShivendraSinha418/XceptionNet_Deepfake_detection_model/blob/main/dfmodel.png)

---

## 🔍 Project Overview

With the rise of AI-generated fake videos, it's becoming increasingly important to develop tools to distinguish real content from synthetic media. This project uses a **fine-tuned XceptionNet model** to detect deepfakes from video input and provides a simple **Flask-based interface** for predictions.

---

## 🚀 Features

- ✅ Input: Video files (`.mp4`, `.avi`, etc.)
- ✅ Fine-tuned **XceptionNet** on real vs fake dataset
- ✅ Frame extraction and preprocessing pipeline
- ✅ Batch-level prediction for robustness
- ✅ Flask interface for local usage
- ❌ Not yet deployed online (can be run locally)
---
## 📊 Dataset

The model is trained on a **custom curated dataset** comprising preprocessed image frames extracted from three popular deepfake datasets:

- 🧑‍🎤 **[Celeb-DF (v2)](https://github.com/yuezunli/Celeb-DF)**  
- 🧪 **[DFDC (Deepfake Detection Challenge)](https://www.kaggle.com/c/deepfake-detection-challenge)**  
- 🎬 **[FaceForensics++](https://github.com/ondyari/FaceForensics)**

These frames were extracted, resized to `299x299`, normalized, and labeled as **real** or **fake** for training the model.

📁 **Kaggle Dataset:**  
👉 [Click here to view/download](https://www.kaggle.com/datasets/shivendrasinha/combined-datasetdfdcceleb-dfff)

> This dataset was uploaded by me and is not included in this repository due to its size. Please refer to the Kaggle link above for downloading and usage.

---

## 🧠 Model Architecture

- **Base Model**: XceptionNet (pre-trained on ImageNet)
- **Transfer Learning**: Last layers fine-tuned on custom real/fake video frame dataset
- **Input Shape**: `299x299x3` RGB frames
- **Classification**: Binary (Real / Fake)

---

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- OpenCV (for video processing)
- Flask (for serving the model)
- NumPy, Pandas

---

## 📁 Project Structure

deepfake-detector/
│
├── static/
│ └── styles.css (optional)
│
├── templates/
│ └── index.html
│
├── model/
│ └── xception_deepfake_model.h5
│
├── uploads/
│ └── (video files uploaded for prediction)
│
├── app.py # Flask app
├── video_utils.py # Frame extraction & preprocessing
├── predict.py # Model loading & prediction
├── requirements.txt
└── README.md

---

## 🧪 How It Works

1. **User Uploads Video**  
   → through the Flask interface

2. **Frame Extraction**  
   → Selects key frames from the video using OpenCV

3. **Preprocessing**  
   → Resizes and normalizes frames to match XceptionNet's input

4. **Prediction**  
   → Each frame is passed through the model  
   → Final label is determined via majority vote or average confidence

5. **Result Display**  
   → Real / Fake label shown in the Flask web app

---

## 🖼️ Sample Interface Screenshot

![Flask Interface Screenshot](https://github.com/ShivendraSinha418/XceptionNet_Deepfake_detection_model/blob/main/dfinterface.png)

---

## ▶️ Running the Project Locally
```
bash
# Step 1: Clone the repository
git clone https://github.com/ShivendraSinha418/XceptionNet_Deepfake_detection_model.git
cd deepfake-detector

# Step 2: Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the Flask app
python app.py
