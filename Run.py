import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import os

def run_inference(model_path, data_yaml):
    cap = cv2.VideoCapture(0)  # Initialize the webcam
    model = YOLO(model_path)  # Load your trained model

    # Load class names from the data.yaml
    with open(data_yaml, 'r') as f:
        data = f.read()
    class_names = [line.strip() for line in data.splitlines() if line.startswith('-')]

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            break

        results = model(frame)  # Perform inference

        # Extract boxes, confidences, and class IDs
        boxes = results[0].boxes.xyxy.numpy()
        confidences = results[0].boxes.conf.numpy()
        class_ids = results[0].boxes.cls.numpy()

        # Draw boxes on the frame using OpenCV
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(class_id)]  # Get the class name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw the box
            cv2.putText(frame, f'{class_name}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('YOLOv8 Inference', frame)  # Display the annotated frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit on 'q'

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Main execution flow
if __name__ == "__main__":
    # ... (previous code)

    # Step 3: Run inference
    run_inference('runs/detect/train/weights/best.pt','D:/objdetection/Human-detector-1/data.yaml')  # Pass the path to your trained model and data.yaml