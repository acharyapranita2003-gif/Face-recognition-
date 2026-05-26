import cv2
import numpy as np
from keras.models import load_model

model = load_model(r"D:\face recognition\ML_Model\Emotion.h5")

emotions =['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap =cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print('webcam is open. Press "q" to quit.')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        preds = model.predict(roi_gray, verbose=0)
        emotion_idx = np.argmax(preds[0])
        emotion_label = emotions[emotion_idx]
        confidence =np.max(preds[0]) * 100

        label_text = f"{emotion_label} ({confidence:.1f}%)"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,200, 255), 2)
        cv2.putText(frame, label_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.program terminated.")
