import cv2
import os

def preprocess_images(input_folder, output_folder, img_size=(224, 224)):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        # Resize and normalize
        if img is not None:
            img_resized = cv2.resize(img, img_size)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, img_resized)
            print(f"Processed: {output_path}")

# Example usage
if __name__ == "__main__":
    preprocess_images("data/captured_faces/1", "data/train/1")
