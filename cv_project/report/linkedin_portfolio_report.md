# Safety Helmet Detection Using YOLOv8

## 1. Project Overview
I built a computer-vision safety system that detects whether workers are wearing safety helmets. Helmet rules are hard to enforce by watching cameras manually, so an automatic detector can flag violations as they happen. The model detects three classes - helmet, no_helmet (a bare head) and person - and for every image or video frame it draws labelled bounding boxes with confidence scores and marks the scene as COMPLIANT or VIOLATION. The goal was to go through the complete AI workflow, from data collection and annotation to training, evaluation, error analysis and a demo.

## 2. Tools and Technologies Used
Python, Google Colab (GPU: Tesla T4), Google Drive, YOLOv8, Ultralytics, OpenCV, Roboflow Annotate, Pandas, Matplotlib, ImageHash, Gradio and Streamlit.

## 3. Dataset Preparation
I started from the public Hard Hat Workers dataset (Kaggle, CC0 license, 5000 images), removed 92 near-duplicate images with perceptual hashing, and curated 500 images so that every class and difficult condition (small, dark, blurry and crowded scenes) is represented. Manual Roboflow batch of 120 images prepared (import pending), following strict annotation guidelines, and audited my labels against the original ones. I converted all annotations from Pascal VOC to YOLO format and created a stratified 70/20/10 split (349 train / 100 validation / 51 test).

## 4. Model Training
I trained a YOLOv8n baseline and a YOLOv8s experiment on Google Colab using pretrained COCO weights, image size 640 and 50 epochs (batch 16). The best model (yolov8n.pt) took 6.9 minutes on a Tesla T4 and was selected using validation mAP; its best checkpoint is saved as best.pt.

## 5. Evaluation Results
On the unseen test set the model reached precision 0.73, recall 0.65, mAP@0.5 0.68 and mAP@0.5:0.95 0.40. The best-performing class was no_helmet and the weakest was person. I also studied the confusion matrix and training/validation loss curves to understand behaviour across classes.

## 6. Inference Results
I tested the model on 51 unseen images: it draws bounding boxes, labels and confidence scores and reports a compliance status for each image. I also ran it on video (300 frames at about 55.8 FPS) and built a web demo where anyone can upload a photo.

## 7. Error Analysis
I matched every prediction with the ground truth and found 140 false negatives, 78 false positives, 6 wrong-class predictions and 7 poor boxes. The most common causes were small object, partially visible at image edge, occlusion / crowded scene. Some "false positives" were actually correct detections of people the original dataset never labelled - a reminder that label quality limits measured accuracy.

## 8. What I Learned
- How to choose, legally source and clean a computer-vision dataset (including duplicate removal)
- Manual bounding-box annotation and writing clear annotation guidelines
- Converting Pascal VOC annotations to the YOLO label format
- Training and comparing YOLOv8 models with transfer learning on Google Colab
- What precision, recall, mAP@0.5 and mAP@0.5:0.95 really measure
- That data quality and consistent labels matter as much as the model
- How systematic error analysis turns mistakes into a concrete improvement plan

## 9. Future Improvements
- Add more training images for the weakest class `person` (test mAP@0.5:0.95 = 0.12) and rebalance classes with targeted collection.
- Small objects caused 98 errors: train at a higher resolution (imgsz=960) or use tiled/SAHI inference for far-away workers.
- Re-annotate the public images with missing labels (especially `person`) using the same guidelines as my manual batch - the audit showed the source labels are incomplete.
- Collect more low-light and motion-blurred site images and enable stronger brightness/blur augmentation.
- Add hard-negative examples of caps, hoods and headscarves labelled `no_helmet` to reduce helmet / no_helmet confusion.

## 10. LinkedIn Post
Today, I completed an end-to-end Computer Vision project as part of the AIRI Team PITB AI Internship Task 1.

I built a Safety Helmet Detection System using YOLOv8 that flags workers without helmets in images and video.

The project covered the complete AI workflow:
- Problem selection
- Data collection (500 curated images, duplicates removed)
- Image annotation in Roboflow
- YOLO-format dataset preparation
- Model training on Google Colab (YOLOv8n vs YOLOv8s)
- Model evaluation (mAP@0.5 = 0.68 on unseen test images)
- Inference on unseen images and video
- Error analysis
- Final documentation

This project helped me understand how real-world Computer Vision systems are developed beyond just training a model. I learned the importance of dataset quality, annotation consistency, evaluation metrics, and error analysis.

Tools used: Python, Google Colab, YOLOv8, Ultralytics, OpenCV, and Roboflow.

Key learning: A good AI model depends not only on the algorithm, but also on clean data, correct labels, proper evaluation, and continuous improvement.

#ComputerVision #DeepLearning #YOLOv8 #ArtificialIntelligence #MachineLearning #Python #AIRI #PITB #AIInternship
