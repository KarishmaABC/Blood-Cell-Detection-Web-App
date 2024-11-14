


import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_model(model_path='C:/Users/karishma/Desktop/OD/datasets/fine_tuned_model/yolov8_bccd3/weights/best.pt'):
    model = YOLO(model_path)
    return model


def detect_objects(model, image):
    # Perform inference on the image
    results = model.predict(source=image, save=False)

    # Extract detection data
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        detections.append((cls_id, confidence))

    return results, detections  # Return both results and detections


def draw_boxes(results, image, model):
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = result.conf[0]
        cls = result.cls[0]
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image


# import torch
# import cv2
# from PIL import Image
# import numpy as np
#
# import cv2
#
# def draw_boxes(image, boxes):
#     for box in boxes:
#         x, y, w, h = box[:4]
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     return image
#
# # Load the fine-tuned model
# def load_model():
#     model = torch.hub.load('ultralytics/yolov8', 'custom', path='models/yolov8.pt')
#     return model
#
# # Run object detection on an uploaded image
# def detect_objects(model, image):
#     results = model(image)
#     detections = results.xyxy[0].numpy()  # Bounding boxes with confidence scores
#     return detections, results  # results contains annotated image as well
# def load_model(model_path):
#     from ultralytics import YOLO
#     return YOLO(model_path)
# import numpy as np
# from ultralytics import YOLO
#
#
# def load_model(model_path):
#     return YOLO(model_path)
#
#
# def detect_objects(model, image):
#     # Run the model prediction
#     results = model(image)
#
#     # Initialize a list to store detections
#     detections = []
#     import cv2
#
#     def draw_boxes(image, detections):
#         for det in detections:
#             x1, y1, x2, y2, conf, cls = map(int, det[:4])  # Bounding box coordinates
#             label = f"Class {cls} Conf: {conf:.2f}"
#             # Draw the bounding box
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             # Put label text above the box
#             cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#         return image
#
#     # Process each result
#     for result in results:
#         # Access bounding boxes in `xyxy` format and confidence scores
#         boxes = result.boxes
#         if boxes is not None:
#             xyxy = boxes.xyxy.numpy()
#             conf = boxes.conf.numpy()
#             cls = boxes.cls.numpy()
#
#             # Stack all detections together
#             detection = np.hstack([xyxy, conf[:, None], cls[:, None]])
#             detections.append(detection)
#
#     # Return the results and detections
#     return results, np.vstack(detections) if detections else np.array([])
