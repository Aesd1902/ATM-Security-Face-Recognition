import cv2
import os
import sqlite3

# Initialize SQLite database
def initialize_database(db_name="atm_users.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            user_name TEXT,
            folder_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_user_to_db(user_id, user_name, folder_path, db_name="atm_users.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (user_id, user_name, folder_path) VALUES (?, ?, ?)',
                   (user_id, user_name, folder_path))
    conn.commit()
    conn.close()

def capture_faces(user_id, user_name, save_dir="data/captured_faces", db_name="atm_users.db"):
    initialize_database(db_name)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a user-specific directory
    user_dir = os.path.join(save_dir, f"{user_id}_{user_name}")
    os.makedirs(user_dir, exist_ok=True)

    # Save metadata to the database
    save_user_to_db(user_id, user_name, user_dir)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    print(f"Capturing face data for {user_name} (ID: {user_id})... Press 'q' to quit.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Display the frame
        cv2.imshow("Capture Face", frame)

        # Save frame to user directory
        img_path = os.path.join(user_dir, f"face_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        count += 1

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {user_name}.")

# Example Usage
if __name__ == "__main__":
    user_id = input("Enter User ID: ")
    user_name = input("Enter User Name: ")
    capture_faces(user_id, user_name)
