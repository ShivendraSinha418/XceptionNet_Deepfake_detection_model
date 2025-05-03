from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = tf.keras.models.load_model('./model/XceptionNet_trained_model.h5')

# Basic preprocessing
def basic_preprocessing(img):
    img = cv2.resize(img, (299, 299))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img.astype(np.float32)
    return img

# Prediction logic on video
def predict_on_video(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frame_preds = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            img = basic_preprocessing(frame)
            img = np.expand_dims(img, axis=0)
            pred = model.predict(img)[0][0]
            frame_preds.append(pred)

        frame_count += 1

    cap.release()

    if frame_preds:
        avg_pred = np.mean(frame_preds)
        label = "Real" if avg_pred > 0.98 else "Fake"
        confidence = f"{avg_pred*100:.2f}%" if label == "Real" else f"{(avg_pred/2)*100:.2f}%"
        return label, confidence
    else:
        return "Unable to read frames", "0%"

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    confidence = None
    video_path = None

    if request.method == 'POST':
        file = request.files['video']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            video_path = filepath
            prediction, confidence = predict_on_video(video_path)

    return render_template('predict.html', prediction=prediction, confidence=confidence, video_path=video_path)

if __name__ == '__main__':
    app.run(debug=True)
