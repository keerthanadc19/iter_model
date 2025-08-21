import cv2

# Load pre-trained model files (MobileNet SSD)
prototxt_path = "deploy.prototxt"
model_path = "mobilenet_iter_73000.caffemodel"

# Download if not present
import os
if not os.path.exists(prototxt_path):
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt", 
        prototxt_path
    )
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel", 
        model_path
    )

# Class labels for MobileNet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:  # confidence threshold
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}",
                        (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
