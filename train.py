import os
import json
import tensorflow as tf

os.makedirs("model", exist_ok=True)

# load training dataset with 20% validation split
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32  # load images in batches of 32 for training efficiency.
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# Get class names before normalization
class_names = train_ds.class_names

# Print class names for verification
print("Class names:", class_names)

# Normalize pixel values [0, 1]
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))


# TensorFlow, you decide the fastest way to load/process data
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Save class names to JSON
with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    # Converts the 2D feature maps into a 1D vector.
    # if a 32×32×64 feature map → Flatten → gives 65536 values in a single line.
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
    # Softmax turns the outputs into probabilities ([0.05, 0.90, 0.05])
])

model.compile(
    optimizer='adam',   #Adam = Adaptive Moment Estimation
    loss='sparse_categorical_crossentropy',
    # Loss function = tells the model how wrong its predictions are
    metrics=['accuracy'] #shows performance in a simple number.
)

# Add callbacks for early stopping and saving best model
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('model/best_model.h5', save_best_only=True)
]

# EarlyStopping → stop training if no improvement.
# ModelCheckpoint → save the best model automatically.

# Train model with validation and callbacks
history = model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

# Print final training and validation accuracy
print("Training accuracy:", history.history['accuracy'][-1])
print("Validation accuracy:", history.history['val_accuracy'][-1])

model.save("model/model.h5")
print(" Model and class names saved successfully.")


# Epoch: One round of training on the entire dataset.
# model.fit() → starts training the model.
