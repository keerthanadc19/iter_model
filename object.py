import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import urllib.request

# ======================
# Download Model if Not Present
# ======================
prototxt_path = "deploy.prototxt"
model_path = "mobilenet_iter_73000.caffemodel"

if not os.path.exists(prototxt_path):
    st.write("📥 Downloading deploy.prototxt...")
    urllib.request.urlretrieve(
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt",
        prototxt_path
    )
if not os.path.exists(model_path):
    st.write("📥 Downloading mobilenet_iter_73000.caffemodel...")
    urllib.request.urlretrieve(
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel",
        model_path
    )

# ======================
# Load Model
# ======================
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# ======================
# Streamlit App
# ======================
st.title("📷 Object Detection with MobileNet SSD")
st.write("Upload an image to detect objects using MobileNet SSD (VOC classes).")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    open_cv_image = np.array(image)
    (h, w) = open_cv_image.shape[:2]

    # Prepare blob
    blob = cv2.dnn.blobFromImage(cv2.resize(open_cv_image, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Draw detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(open_cv_image, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(open_cv_image, f"{label}: {confidence:.2f}",
                        (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    st.image(open_cv_image, channels="RGB", caption="Detected Objects")
