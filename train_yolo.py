from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8n, yolov8s, yolov8m, etc. depending on resources

# Train the model
model.train(
    data='data.yaml',          # Path to your data config file
    epochs=50,                  # Number of training epochs
    imgsz=416,                  # Image size
    batch=16,                   # Batch size
    project='fine_tuned_model', # Project name to save results
    name='yolov8_bccd',         # Name of training run
    pretrained=True             # Start from a pretrained model
)
