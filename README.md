# Animal Image Classification using Deep Learning

This project builds a deep learning model that can classify images of animals into one of **15 categories**.  
The dataset consists of colored RGB images sized **224×224×3**, organized into folders representing their class labels.

Unlike earlier projects (Heart Disease, Liver Cirrhosis, Forest Cover), this project deals with **unstructured image data**, and uses **Convolutional Neural Networks (CNNs) and Transfer Learning** instead of classical machine-learning models.

---

## Problem Overview

The dataset contains images belonging to the following **15 animal classes**:

- Bear  
- Bird  
- Cat  
- Cow  
- Deer  
- Dog  
- Dolphin  
- Elephant  
- Giraffe  
- Horse  
- Kangaroo  
- Lion  
- Panda  
- Tiger  
- Zebra  

## Approach & Methodology

### **1. Data Loading & Preprocessing**
- Loaded 15-class dataset from directory structure  
- Applied PyTorch transforms:
  - Resizing to **224×224**
  - Random flips & rotations
  - Normalization (ImageNet stats)
- Created train / validation loaders  

### **2. Model Architecture**
This project uses **Transfer Learning** with **ResNet50**, a powerful CNN pretrained on ImageNet.

- Replaced the final fully-connected layer to output 15 classes  
- Fine-tuned the model on the custom dataset  

### **3. Training Setup**
- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Learning rate: 1e-4  
- Batch size: 32  
- Epochs: 10  
- GPU acceleration used (CUDA available)  

---

## Model Performance

### **Training & Validation Metrics per Epoch**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|------|------------|-----------|----------|---------|
| 1 | 2.4879 | 29.07% | 2.2218 | 54.76% |
| 2 | 2.0219 | 69.90% | 1.8055 | 82.78% |
| 3 | 1.6412 | 85.72% | 1.4705 | 87.66% |
| 4 | 1.3626 | 89.07% | 1.2266 | 91.00% |
| 5 | 1.1375 | 90.42% | 1.0457 | 89.20% |
| 6 | 0.9584 | 92.99% | 0.9149 | 91.26% |
| 7 | 0.8378 | 93.12% | 0.7671 | 93.06% |
| 8 | 0.7440 | 94.34% | 0.7213 | 92.03% |
| 9 | 0.6714 | 94.15% | 0.6584 | 91.77% |
| 10 | 0.6055 | 94.66% | 0.5649 | 93.06% |

---

### **Final Validation Accuracy**
- The model shows strong learning progression.  
- Achieves **93% accuracy** on unseen validation images.  
- Indicates successful transfer learning using ResNet50.
