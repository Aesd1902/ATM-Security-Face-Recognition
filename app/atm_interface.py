import cv2
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import random

DB_NAME = "atm_users.db"
TRANSACTION_HISTORY = "transaction_history.txt"

# Load user metadata
def load_users_from_db(db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('SELECT user_id, user_name, folder_path FROM users')
    users = [{"user_id": row[0], "user_name": row[1], "folder": row[2]} for row in cursor.fetchall()]
    conn.close()
    return users

# Recognize face
def recognize_face(model, frame, users, threshold=0.7):
    resized_frame = cv2.resize(frame, (224, 224))
    array_frame = img_to_array(resized_frame)
    processed_frame = preprocess_input(np.expand_dims(array_frame, axis=0))

    prediction = model.predict(processed_frame)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    if confidence > threshold:
        user = users[predicted_class]
        return user, confidence
    return None, confidence

# Liveness detection placeholder
def liveness_detection():
    # Simulate a liveness detection process (e.g., blink detection)
    print("Performing liveness detection...")
    return random.choice([True, False])  # Simulate success or failure

# ATM transaction interface
def atm_transaction_interface(user):
    print(f"Welcome {user['user_name']}! Proceeding to ATM transaction...")
    print("Available options: 1. Withdraw 2. Check Balance 3. Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        amount = input("Enter withdrawal amount: ")
        print(f"Withdrawing {amount}...")
        log_transaction(user, "Withdraw", amount)
    elif choice == "2":
        print("Checking balance...")
        log_transaction(user, "Check Balance", "N/A")
    else:
        print("Exiting transaction.")
        log_transaction(user, "Exit", "N/A")

# Log transaction
def log_transaction(user, action, amount):
    with open(TRANSACTION_HISTORY, "a") as f:
        f.write(f"{user['user_id']},{user['user_name']},{action},{amount}\n")
    print(f"Transaction logged: {action} - {amount}")

# Multi-factor authentication
def multi_factor_auth():
    otp = random.randint(100000, 999999)
    print(f"Your OTP is: {otp}")
    entered_otp = input("Enter the OTP: ")
    return str(otp) == entered_otp

# Main ATM interface
if __name__ == "__main__":
    # Load the face recognition model
    model = load_model("models/face_recognition_model.h5")

    # Load registered user data
    users = load_users_from_db()
    if not users:
        print("No registered users found. Please capture face data first.")
        exit()

    # Initialize webcam for face recognition
    cap = cv2.VideoCapture(0)
    print("ATM Face Recognition Started. Please look at the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.imshow("ATM Face Recognition", frame)

        # Perform face recognition
        user, confidence = recognize_face(model, frame, users)
        if user:
            print(f"Face Recognized: {user['user_name']} (Confidence: {confidence:.2f})")
            if liveness_detection():
                print("Liveness detection passed.")
                if multi_factor_auth():
                    print("Multi-factor authentication successful.")
                    atm_transaction_interface(user)
                else:
                    print("Multi-factor authentication failed.")
            else:
                print("Liveness detection failed.")
            break

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting ATM interface...")
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
