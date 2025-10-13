** Object Detection Project **

Description:
This project implements an Object Detection system in Python using machine learning and computer vision techniques. It can detect and classify multiple objects in images or videos, draw bounding boxes around them, and optionally save the output. The project supports real-time detection via webcam and uses pretrained models for accurate recognition.


Key Features:

Detects objects in images
Draws bounding boxes with labels
Real-time detection with webcam input
Uses pretrained models for high accuracy


** Testing  **

I test my model on a separate dataset the model hasn’t seen before. I preprocess the images, make predictions, and compare them with true labels to calculate overall and class-wise accuracy. This shows how well the model generalizes and helps identify classes it struggles with.



Technologies Used:

Python 3.12
OpenCV
TensorFlow / PyTorch (for object detection models)
Numpy
