from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO("runs/detect/yolov8-oyster/weights/best.pt")

# Open the webcam and start real-time prediction
model.predict(
    source=0,        # 0 = webcam (use 1 or 2 if multiple cameras are connected)
    show=True,       # Show the video with bounding boxes
    conf=0.4,        # Minimum confidence threshold for detections
    save=False       # Do not save the output images
)
