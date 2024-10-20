import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Define ImageDataGenerator with rescaling and augmentation for training
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% for validation
)

# Load training and validation data
train_generator = datagen.flow_from_directory(
    './dataset',  # Change to your dataset path
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    './dataset',  # Change to your dataset path
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Load pre-trained EfficientNetB0 without the top layer
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model initially

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

# Create and compile the model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=validation_generator.samples // BATCH_SIZE
)


import numpy as np

def estimate_shelf_life(model, image_path, baseline_shelf_life):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get the prediction (probability of being fresh)
    freshness_prob = model.predict(img_array)[0][0]  # Since output is sigmoid, this is the probability of being "fresh"

    # Start with 50% rotten assumption
    initial_rot = 0.5
    rot_percent = initial_rot - (freshness_prob - initial_rot)  # Adjusted based on freshness probability

    # Calculate estimated shelf life
    estimated_shelf_life = baseline_shelf_life * (1 - rot_percent)

    return estimated_shelf_life, freshness_prob

# Example usage
baseline_shelf_life = 10  # Assuming 10 days as the base shelf life for the fruit type
image_path = 'path_to_sample_image'  # Path to a sample image
shelf_life, freshness_prob = estimate_shelf_life(model, image_path, baseline_shelf_life)

print(f"Estimated Shelf Life: {shelf_life} days")
print(f"Freshness Probability: {freshness_prob * 100:.2f}%")

