from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os

def train_model(data_dir, model_save_path):
    # Load pre-trained model
    base_model = MobileNetV2(weights="imagenet", include_top=False)
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    predictions = Dense(len(os.listdir(data_dir)), activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    # Data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2)
    train_generator = train_datagen.flow_from_directory(data_dir, target_size=(224, 224),
                                                        batch_size=32, class_mode="categorical")
    
    # Train model
    model.fit(train_generator, epochs=10, verbose=1)
    model.save(model_save_path)
    print(f"Model saved at: {model_save_path}")

# Example usage
if __name__ == "__main__":
    train_model("data/captured_faces", "models/face_recognition_model.h5")
