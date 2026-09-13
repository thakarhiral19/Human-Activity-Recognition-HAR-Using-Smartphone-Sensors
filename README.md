# 🏃‍♂️ Human Activity Recognition (HAR) using Smartphone Sensors

[![Live Demo](https://img.shields.io/badge/Live%20App-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)](https://humanactivityrecognitionusingsmartphonesensors.streamlit.app/)

## 📌 Project Overview
This Data Science and Machine Learning project predicts a person's physical activity based on data collected from smartphone built-in sensors (Accelerometer and Gyroscope). 

The web application allows users to upload raw sensor data (CSV format) and instantly predicts the activity being performed using a trained **Random Forest Classifier** model.

**🔗 [Click here to view the Live Web App](https://humanactivityrecognitionusingsmartphonesensors.streamlit.app/)**

## 🧠 Machine Learning Model
- **Algorithm:** Random Forest Classifier (100 estimators)
- **Features:** 561 extracted features from time and frequency domain variables.
- **Accuracy:** ~91.14% on unseen test data.
- **Classes Predicted (6):** 
  - Walking 🚶
  - Walking Upstairs 🧗
  - Walking Downstairs 🚶‍♂️
  - Sitting 🪑
  - Standing 🧍
  - Laying 🛌

## 🛠️ Tech Stack
- **Frontend & Backend UI:** Streamlit
- **Machine Learning:** Scikit-Learn
- **Data Manipulation:** Pandas, NumPy
- **Model Serialization:** Joblib

## 📊 Features of the Web App
1. **CSV Upload:** Users can upload a CSV file containing 561 sensor readings.
2. **Sensor Waveform Visualization:** Automatically plots a line chart of the raw sensor values.
3. **Real-time Prediction:** Instantly predicts the activity based on the uploaded data.
4. **Confidence Score Analysis:** Displays a bar chart showing the probability distribution across all 6 activities, indicating how confident the model is in its prediction.

## 💻 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/HAR-Project.git](https://github.com/YOUR_GITHUB_USERNAME/HAR-Project.git)
