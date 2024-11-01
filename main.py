import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Predefined parking zones (manually marked)
# (800,168,50,100),
parking_zones = [(0,160,840,221),(0,314,840,466)]  # Example parking spots
parking_times = {}  # Dictionary to store entry times of vehicles
illegal_parking_limit = 5  # Time limit in seconds for illegal parking

cap = cv2.VideoCapture("illegal.mp4")  # Use 0 for webcam or a video file

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)


    # Initialize list of detected cars
    detected_vehicles = []
    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Only detect cars (COCO class 'car' has id 2, 'truck' is 7)
            if confidence > 0.6 and class_id in [1,2,3,7]:  # 'car' or 'truck'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                # detected_vehicles.append((x, y, x + w, y + h))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            detected_vehicles.append((x, y, x + w, y + h))

    for zone_idx, (x1, y1, x2, y2) in enumerate(parking_zones):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw parking zones
        
        for (startX, startY, endX, endY) in detected_vehicles:
            # cenX,cenY = (startX+endX)//2,(startY+endY)//2
            # if  x1 > cenX > x2 and y1 > cenY > y2:
            if startX > x1 and endX < x2 and startY > y1 and endY < y2:
                vehicle_id = f"vehicle_{zone_idx}"
                
                # Mark entry time for new vehicle
                if vehicle_id not in parking_times:
                    parking_times[vehicle_id] = time.time()
                else:
                    # Check duration vehicle has been in the parking zone
                    parked_time = time.time() - parking_times[vehicle_id]
                    if parked_time > illegal_parking_limit:
                        cv2.putText(frame, "ILLEGAL PARKING", (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Reset time if vehicle leaves the parking zone
                if f"vehicle_{zone_idx}" in parking_times:
                    del parking_times[f"vehicle_{zone_idx}"]

            # Draw bounding box around detected vehicle
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
    
    # Display the output frame
    cv2.imshow('Parking Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
