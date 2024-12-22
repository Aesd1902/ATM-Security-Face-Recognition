from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_liveness_model(data_dir, model_save_path):
    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid")  # Binary classification (Live/Not Live)
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    # Data augmentation
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(data_dir, target_size=(224, 224),
                                            batch_size=32, class_mode="binary", subset="training")
    val_gen = datagen.flow_from_directory(data_dir, target_size=(224, 224),
                                          batch_size=32, class_mode="binary", subset="validation")
    
    # Train model
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save(model_save_path)
    print(f"Liveness model saved at: {model_save_path}")

# Example usage
if __name__ == "__main__":
    train_liveness_model("data/casia_liveness", "models/liveness_detection_model.h5")
