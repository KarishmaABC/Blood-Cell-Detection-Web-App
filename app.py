


# app.py
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
# from utils import load_model, detect_objects, draw_boxes
from custom_utils import load_model, detect_objects, draw_boxes

st.title("Object Detection Web App")
st.write("Upload an image to detect objects using BCCD")


@st.cache_resource
def get_model():
    return load_model('C:/Users/karishma/Desktop/OD/datasets/fine_tuned_model/yolov8_bccd3/weights/best.pt')
# models/yolov8n.pt
# from custom_utils import load_model, detect_objects, draw_boxes

# Load model, detect objects, and draw boxes


model = get_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Processing...")

    image_np = np.array(image)
    results, detections = detect_objects(model, image_np)  # Get both results and detections

    # Draw bounding boxes
    image_with_boxes = draw_boxes(results, image_np, model)
    st.image(image_with_boxes, caption='Detected Objects', use_container_width=True)
    st.write("Detection complete!")

    # Prepare data for the detection table
    detection_data = []
    for cls_id, confidence in detections:
        detection_data.append({
            'Class': model.names[cls_id],
            'Confidence Score': f'{confidence:.2f}'
        })

    # Display table
    st.write("Detection Confidence Scores")
    st.table(pd.DataFrame(detection_data))
