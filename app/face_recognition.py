import cv2
import numpy as np
from tensorflow.keras.models import load_model

def recognize_face(frame, model):
    # Preprocess frame
    img = cv2.resize(frame, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    predictions = model.predict(img)
    class_id = np.argmax(predictions)
    confidence = predictions[0][class_id]
    
    return class_id, confidence

# Load trained model
model = load_model("models/face_recognition_model.h5")

# Example usage with webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    class_id, confidence = recognize_face(frame, model)
    cv2.putText(frame, f"ID: {class_id}, Conf: {confidence:.2f}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
