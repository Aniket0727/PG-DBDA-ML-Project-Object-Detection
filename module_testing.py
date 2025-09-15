import tensorflow as tf
import numpy as np
import json
import os

MODEL_PATH = "model/model.h5"
CLASS_NAMES_PATH = "model/class_names.json"
TEST_DIR = "data"  


model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


# Loading and preprocessing test dataset
# loads the test dataset from a directory

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Normalize images
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))


# Evaluate overall accuracy
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


y_true = np.array(y_true)
y_pred = np.array(y_pred)

overall_accuracy = np.mean(y_true == y_pred)
print(f"Overall Test Accuracy: {overall_accuracy*100:.2f}%")



print("\nClass-wise Accuracy:")
for idx, class_name in enumerate(class_names):

    true_count = np.sum(y_true == idx)
    
    correct_count = np.sum((y_true == idx) & (y_pred == idx))
    
    if true_count > 0:
        class_acc = correct_count / true_count * 100
        print(f"{class_name}: {class_acc:.2f}% ({correct_count}/{true_count})")
    else:
        print(f"{class_name}: No samples in test set")

