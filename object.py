import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import urllib.request

# =========================
# Download Model Files if not present
# =========================
MODEL_URL = "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/MobileNetSSD_deploy.prototxt"


MODEL_PATH = "MobileNetSSD_deploy.caffemodel"
PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading MobileNetSSD model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

if not os.path.exists(PROTOTXT_PATH):
    st.write("📥 Downloading MobileNetSSD config...")
    urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)

# =========================
# Load Pretrained Model
# =========================
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

# Class labels for the model (VOC 20 classes + background)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Assign random colors to classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# =========================
# Streamlit App
# =========================
st.title("📷 Object Detection with MobileNet SSD")
st.write("Upload an image and detect objects using MobileNet-SSD pretrained model (auto-downloads model files).")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    open_cv_image = np.array(image)
    (h, w) = open_cv_image.shape[:2]

    # Prepare input blob for MobileNet SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(open_cv_image, (300, 300)), 
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Draw detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            color = COLORS[idx]

            cv2.rectangle(open_cv_image, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(open_cv_image, label, (startX, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    st.image(open_cv_image, channels="RGB", caption="Detected Objects")
