# Object_Detection_Web_App
This project is an interactive web application built using Streamlit to detect blood cells in images. It utilizes a fine-tuned YOLOv8 object detection model, trained on the BCCD (Blood Cell Count and Detection) dataset to recognize and classify different types of blood cells.

Table of Contents

Project Overview

Features

Setup and Installation

Dataset

Model Training

Usage

Future Enhancements

References

Project Overview

The Object Detection Web App is a user-friendly interface that allows users to upload images and detect different blood cell types, including RBCs (Red Blood Cells), WBCs (White Blood Cells), and Platelets. The app is useful for educational purposes and healthcare applications where quick blood cell analysis is required.

Features
Image Upload: Users can upload images of blood smears.

Real-Time Detection: The app processes the image using a fine-tuned YOLOv8 model to detect and label blood cells.

Interactive Visualization: Detected blood cells are marked with bounding boxes for easy identification.

Confidence Scores: Each detection shows a confidence score, indicating the model’s certainty in the prediction.

Setup and Installation

Prerequisites

Python 3.8+
Streamlit for building the web interface.

Ultralytics YOLOv8 for object detection.

OpenCV for image processing.

Torch for PyTorch-based model loading and inference.

Installation Steps

Clone the Repository

git clone https://github.com/KarishmaABC/object-detection-web-app
cd object-detection-web-app
Install Dependencies Make sure you are in a virtual environment, then install the required packages:


pip install -r requirements.txt

Download the Fine-Tuned Model

Place your fine-tuned YOLOv8 model (e.g., yolov8n_bccd.pt) in the models/ directory. You can train this model on the BCCD Dataset.

Streamlit App

Run the Streamlit app using:


streamlit run app.py

The app will launch in your default web browser, typically at http://localhost:8501.

Dataset

The BCCD (Blood Cell Count and Detection) dataset contains annotated images of blood cells, with labels for RBCs, WBCs, and Platelets. This dataset is useful for training and testing blood cell detection models.

Download Link: BCCD Dataset on GitHub

Model Training

To fine-tune the YOLOv8 model on the BCCD dataset, follow these steps:

Prepare the Dataset:

Clone the BCCD dataset repository and organize the images and annotations in YOLO format.

Train the Model:

Use the following command with Ultralytics YOLOv8 to train on the BCCD dataset:


yolo task=detect mode=train model=yolov8n.pt data=config/bccd.yaml epochs=100 imgsz=640
Replace yolov8n.pt with your YOLOv8 model variant (e.g., yolov8s.pt for a larger model).
Save the Model:

Once training is complete, save the fine-tuned model (e.g., yolov8n_bccd.pt) in the models/ directory.









![Screenshot (34)](https://github.com/user-attachments/assets/67f92b84-b7f2-452e-b552-739b536774ad)







![Screenshot (35)](https://github.com/user-attachments/assets/3a5feaaf-cd2f-4116-9be2-c1a24c5dcb61)







Usage
Upload an Image: Click on the file uploader to upload an image of a blood smear.

Detection: Once the image is uploaded, the app will automatically detect and display blood cells, marking them with bounding boxes.
Confidence Scores: Each detection is accompanied by a confidence score, indicating the model’s prediction confidence.

Example Workflow

Start the Streamlit server.
Upload an image of a blood sample.

View the detection results, which include bounding boxes around each cell type.

Future Enhancements
Live Video Feed: Extend support for real-time blood cell detection via live video feed.

Cell Count Summary: Provide a summary of the counts for each type of blood cell.

Export Results: Allow users to download annotated images or a report of detection results.

References

BCCD Dataset

Ultralytics YOLOv8 Documentation

Streamlit Documentation
