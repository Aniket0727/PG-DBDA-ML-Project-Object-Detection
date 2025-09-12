Dataset Loading & Preprocessing

The dataset is stored in a folder structure where each subfolder represents a class (e.g., cats/, dogs/).

The dataset is split into training (80%) and validation (20%) sets.

Images are resized to 128x128 pixels and normalized to the range [0,1] for faster training.

TensorFlow’s AUTOTUNE is used for efficient prefetching and parallel loading.

Model Architecture (CNN)

Conv2D + ReLU → Extracts low-level features like edges.

MaxPooling2D → Reduces image size while keeping important details.

Conv2D + ReLU (deeper) → Extracts more complex patterns.

MaxPooling2D → Again reduces dimensionality.

Flatten → Converts 2D features into 1D vector.

Dense (128, ReLU) → Fully connected layer to learn feature combinations.

Dense (Softmax) → Final layer that outputs class probabilities.

Training

Uses Adam optimizer for adaptive learning.

Sparse categorical crossentropy as the loss function.

Callbacks:

EarlyStopping → Stops training if validation loss doesn’t improve.

ModelCheckpoint → Saves the best model (best_model.h5).

Saving Outputs

The trained model is saved (model.h5).

Class names are stored in class_names.json for easy prediction later.

Prediction

A new image can be loaded, preprocessed, and passed to the model.

The model outputs probabilities, and the highest one determines the predicted class.

🔹 Purpose of the Project

To automatically classify images into different categories (cats, dogs, etc.).

Showcases skills in TensorFlow, deep learning, and data preprocessing.

Can be extended with transfer learning or deployed as a web app.
