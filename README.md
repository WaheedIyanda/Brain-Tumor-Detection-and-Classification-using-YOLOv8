## **Overview**
**Objective:**  
This project aims to develop a robust deep learning model for the detection and classification of brain tumors from MRI images. Using the YOLOv8 object detection framework, the model identifies and classifies three brain tumor types: **Glioma**, **Meningioma**, and **Pituitary tumors**.

---
web app :  https://huggingface.co/spaces/WaheedIyanda/Brain_tumour_detector_and_classifier
---

## **Dataset**
**Dataset Name:** Brain Tumor MRI Dataset  
**Source:** Roboflow Universe  
**Total Images:** 3,903 MRI scans  

**Classes:**
- **Glioma:** Tumors originating from glial cells  
- **Meningioma:** Tumors arising from the meninges  
- **Pituitary Tumor:** Tumors located in the pituitary gland  
- **No Tumor:** MRI scans without tumor presence *(not the main focus)*  

**Annotation Format:**  
YOLO format with images and corresponding `.txt` bounding box label files.

---

## **Dataset Distribution (Training Set)**

| Class | Label | Count | Approx % |
|------|------|------|----------|
| Glioma | 0 | 983 | 45.85% |
| Meningioma | 1 | 503 | 23.46% |
| Pituitary | 2 | 658 | 30.69% |

---

## **Data Splits**
- **Train Images:** 2,144  
- **Validation Images:** 612  
- **Test Images:** 308  

---

## **Methodology**

### **1. Data Loading and Preprocessing**
Dataset loaded from Google Drive, extracted from `archive.zip`, and organized into training, validation, and testing directories.

### **2. Data Exploration and Annotation Verification**
Image counts and class distribution analyzed. Sample images visualized with bounding boxes to verify annotation correctness.

### **3. Model Training**
**Model:** YOLOv8n (YOLOv8 Nano)  
**Initialization:** Pre-trained weights (`yolov8n.pt`)  
**Approach:** Fine-tuned on the brain tumor dataset.

**Training Parameters:**
- **Model:** yolov8n.pt  
- **Data Config:** `/content/archive/BrainTumor/BrainTumorYolov8/data.yaml`  
- **Epochs:** 50  
- **Image Size:** 640 × 640  
- **Batch Size:** 16  

### **4. Model Evaluation and Testing**
Model evaluated on the validation set and tested on unseen data. Detection results visualized using `model.predict()`.

---

## **Performance**
**Validation Metrics:**
- **mAP@50:** 0.919  
- **mAP@50–95:** 0.718  

**Per-Class mAP@50–95:**
- **Glioma:** 0.563  
- **Meningioma:** 0.836  
- **Pituitary:** 0.753  


