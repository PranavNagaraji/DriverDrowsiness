import cv2
import numpy as np
import RPi.GPIO as GPIO
from tensorflow.keras.models import load_model
import time

# Load model and classifiers
model = load_model('model.keras')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# GPIO setup
LED_PIN = 17
BUZZER_PIN = 27
VIBRATION_PIN = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(VIBRATION_PIN, GPIO.OUT)

def signal_alert(on):
    GPIO.output(LED_PIN, on)
    GPIO.output(BUZZER_PIN, on)
    GPIO.output(VIBRATION_PIN, on)

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
        
        drowsy_detected = False

        for (x, y, w, h) in faces[:15]:
            face_region = gray_img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_region)
            
            for (ex, ey, ew, eh) in eyes:
                if ey > face_region.shape[0] // 2:
                    continue
                eye = face_region[ey:ey+eh, ex:ex+ew]
                eye = cv2.resize(eye, (64, 64))
                eye = cv2.cvtColor(eye, cv2.COLOR_GRAY2RGB)
                eye = eye / 255.0
                eye = np.expand_dims(eye, axis=0)

                prediction = model.predict(eye)[0][0]
                label = "Drowsy" if prediction < 0.5 else "Non_Drowsy"
                color = (0, 0, 255) if prediction < 0.5 else (0, 255, 0)

                if prediction < 0.5:
                    drowsy_detected = True

                cv2.putText(frame, f'{label} ({prediction:.2f})', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)

        signal_alert(drowsy_detected)

        cv2.imshow("Live Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
