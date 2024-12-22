import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the liveness detection model
liveness_model = load_model("models/liveness_detection_model.h5")

def detect_liveness(frame):
    # Preprocess the frame for liveness detection
    img = cv2.resize(frame, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict liveness
    prediction = liveness_model.predict(img)
    liveness_score = prediction[0][0]
    
    return liveness_score > 0.5, liveness_score

# Example usage with webcam
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check liveness
        is_live, score = detect_liveness(frame)
        label = "Live" if is_live else "Fake"
        
        # Display results
        cv2.putText(frame, f"Liveness: {label} ({score:.2f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_live else (0, 0, 255), 2)
        cv2.imshow("Liveness Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
