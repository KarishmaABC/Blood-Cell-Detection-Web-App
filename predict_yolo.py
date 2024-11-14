
from ultralytics import YOLO

# Loading the fine-tuned model
model = YOLO('C:/Users/karishma/Desktop/OD/datasets/fine_tuned_model/yolov8_bccd3/weights/best.pt')

# Making predictions
results = model('C:/Users/karishma/Desktop/OD/datasets/BCCD_dataset/test/images/BloodImage_00204_jpg.rf.0555bc62812f0987a35f05f0960dd7c4.jpg')
for result in results:
    result.show()
# Display the result
