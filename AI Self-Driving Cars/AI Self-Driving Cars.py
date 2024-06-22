import cv2
import numpy as np

# Load pre-trained model and configuration file for object detection
net = cv2.dnn.readNetFromCaffe('path_to_deploy.prototxt', 'path_to_model.caffemodel')

def detect_objects(image):
    (h, w) = image.shape[:2]
    # Prepare the image for the deep learning model
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    objects_detected = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Filter out weak detections
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Store the detection result
            objects_detected.append({
                "object_id": idx,
                "confidence": confidence,
                "bounding_box": (startX, startY, endX, endY)
            })

            # Draw the bounding box on the image
            label = f"ID: {idx}, Conf: {confidence:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, objects_detected

# Load an example image
image_path = 'path_to_example_image.jpg'
image = cv2.imread(image_path)

# Detect objects in the image
processed_image, objects_detected = detect_objects(image)

# Display the processed image with detected objects
cv2.imshow("Object Detection", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
