import cv2
from ultralytics import YOLO
from roboflow import Roboflow
import os

# Function to download the dataset from Roboflow
def download_dataset(api_key, project_name, version_number):
    rf = Roboflow(api_key=api_key)
    project = rf.project(project_name)
    version = project.version(version_number)
    dataset = version.download("yolov8")
    print(f"Dataset downloaded to: {dataset.location}")
    
    # Update the data.yaml file paths
    yaml_path = os.path.join(dataset.location, 'data.yaml')
    with open(yaml_path, 'r') as f:
        data = f.read()

    data = data.replace('train: Human-detector-1/train/images', 'train: train/images')
    data = data.replace('val: Human-detector-1/valid/images', 'val: valid/images')

    with open(yaml_path, 'w') as f:
        f.write(data)
    
    return dataset.location

# Function to train the YOLOv8 model
def train_model(dataset_path):
    model = YOLO('yolov8n.pt')  # Load a pre-trained model
    model.train(data=f'{dataset_path}/data.yaml', epochs=50, imgsz=640)  # Adjust epochs and image size as needed

# Function to run inference using the trained model
def run_inference(model_path):
    cap = cv2.VideoCapture(0)  # Initialize the webcam
    model = YOLO(model_path)  # Load your trained model

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw the box
            cv2.putText(frame, f'ID: {int(class_id)}, Conf: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow('YOLOv8 Inference', frame)  # Display the annotated frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Exit on 'q'

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Main execution flow
if __name__ == "__main__":
    # Replace these with your actual values
    API_KEY = "cF1WLN6vHInraHBJCcmJ"  # Your Roboflow API key
    PROJECT_NAME = "human-detector-6mwxx"  # Your project name
    VERSION_NUMBER = 1  # Version of the dataset

    # Step 1: Download the dataset
    try:
        dataset_path = download_dataset(API_KEY, PROJECT_NAME, VERSION_NUMBER)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        exit(1)  # Exit if dataset download fails

    # Step 2: Train the model
    train_model(dataset_path)  # Pass the path to the downloaded dataset

    # Step 3: Run inference
    run_inference('runs/detect/train/weights/best.pt')  # Update with the path to your trained model
