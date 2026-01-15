# 🚗 Driver Drowsiness Detection System

A **real-time Driver Drowsiness Detection System** developed using **Python and OpenCV**.
This project detects driver fatigue by monitoring eye closure and facial features through a live camera feed. If drowsiness is detected for a sustained period, the system raises an alert to help prevent road accidents.

---

## 🎯 Objective

Driver fatigue is one of the major causes of road accidents.
This project aims to:

* Detect driver drowsiness in real time
* Alert the driver before a dangerous situation occurs
* Provide a lightweight and extendable computer vision solution

---

## ✨ Features

* Real-time face and eye detection
* Live webcam-based monitoring
* Raspberry Pi camera support
* Uses Haar Cascade classifiers
* Modular Python codebase
* Can be extended with alarms, buzzers, or notifications

---

## 🧠 How It Works

1. Captures live video frames from a webcam or Pi camera
2. Detects the driver's face using Haar Cascade classifier
3. Detects eyes within the face region
4. Determines eye state (open / closed)
5. Tracks duration of eye closure
6. If eyes remain closed beyond a threshold, drowsiness is detected
7. An alert can be triggered to warn the driver

---

## 🗂️ Project Structure

```
DriverDrowsiness/
│
├── haarcascade_eye.xml
├── haarcascade_frontalface_default.xml
├── live_detect.py
├── liveDetectionPi.py
├── model.py
├── preprocessing.py
├── testing.py
├── model.zip
└── README.md
```

---

## 📁 File Description

* **haarcascade_eye.xml**
  Haar Cascade classifier for eye detection

* **haarcascade_frontalface_default.xml**
  Haar Cascade classifier for face detection

* **live_detect.py**
  Main script for real-time drowsiness detection using a webcam

* **liveDetectionPi.py**
  Detection script designed for Raspberry Pi camera

* **model.py**
  Loads and manages the trained model

* **preprocessing.py**
  Handles image preprocessing and feature extraction

* **testing.py**
  Used for testing the detection logic

* **model.zip**
  Pre-trained model (extract before running)

---

## 🛠️ Requirements

* Python 3.7 or higher
* Webcam or Raspberry Pi Camera

### Python Libraries

```bash
pip install opencv-python numpy tensorflow keras
```

---

## ▶️ How to Run

### Webcam Version

```bash
python live_detect.py
```

### Raspberry Pi Version

```bash
python liveDetectionPi.py
```

Make sure the Haar cascade XML files and extracted model files are in the same directory.

---

## 📌 Notes

* Ensure proper lighting for accurate detection
* Camera should clearly capture the driver's face
* Sensitivity and thresholds can be tuned in the code

---

## 🚀 Future Improvements

* Add sound or buzzer alerts
* Integrate deep learning–based eye landmark detection
* Add performance metrics (accuracy, FPS)
* Build a dashboard for monitoring driver alertness

---

## 📄 License

This project is open-source and free to use for educational and research purposes.

---

## 👨‍💻 Author

**Pranav Nagaraji**
Driver Drowsiness Detection Project
