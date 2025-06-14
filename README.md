# Sign_Language_Detection

## Hand Gesture Classification with OpenCV & TensorFlow

### 📌 Overview:- 

This project uses OpenCV, cvzone, and TensorFlow to detect and classify hand gestures using a pre-trained model (keras_model.h5). The script captures hand images via the device camera, processes them, and predicts the corresponding hand gesture.

### 🛠 Requirements:-

Ensure you have the following dependencies installed:

bash
`pip install opencv-python cvzone numpy tensorflow`


### 🚀 How to Run:-

1️⃣ Clone the repository (if applicable):

bash
`git clone https://github.com/your-repo-name.git`
`cd your-repo-name`


2️⃣ Run the script:

bash
`python filename.py`


### 📷 How It Works:- 

- Hand detection: Uses cvzone.HandTrackingModule.HandDetector to find the hand in the frame.

- Hand cropping & resizing: Extracts the bounding box and ensures the hand fits a fixed-size image (300x300 pixels).

- Gesture classification: Uses cvzone.ClassificationModule.Classifier to predict the gesture label from a trained TensorFlow model.

### 📌 Features
✔ Detects hand gestures using OpenCV. ✔ Classifies gestures using a pre-trained TensorFlow model. ✔ Supports real-time recognition with a webcam.

### 🖼 Example Output
The program captures a hand gesture and overlays bounding boxes & predictions:

#### plaintext
`Predicted Gesture: Hello`


### 🔹 Dependencies:-
- OpenCV → Image processing

- cvzone → Hand tracking & classification

- TensorFlow → Model inference

### ⚡ Next Steps:-

✔ Improve gesture accuracy with more training data. 

✔ Add new gestures for better recognition. 

✔ Optimize performance using GPU acceleration.
