Here is a GitHub README file written in Markdown format for your **Automatic Number Plate Recognition (ANPR) System** project:

---

# Automatic Number Plate Recognition (ANPR) System 🚗

### An efficient system for real-time vehicle license plate detection and recognition.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Use Cases](#use-cases)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [Dataset](#dataset)
- [Training and Testing](#training-and-testing)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

---

## Introduction 📖

The **ANPR System** uses **Deep Learning** and **Computer Vision** to detect and recognize vehicle license plates in real-time. It combines object detection for plate localization with **Optical Character Recognition (OCR)** for extracting the text.

The project implements two models:
1. **Transfer Learning** using TensorFlow Object Detection API for a lightweight model.
2. **RPNet** (custom model) based on a Convolutional Neural Network (CNN) for accurate detection and recognition.

---

## Features ✨

- Real-time license plate detection and recognition.
- Optimized lightweight model for quick deployment.
- High-accuracy model (RPNet) for law enforcement and detailed analysis.
- Supports plate localization, segmentation, and character recognition.
- Transfer learning for faster model training.

---

## Use Cases 🚦

This ANPR System can be used for:
- **Law Enforcement**: Vehicle identification and traffic rule monitoring.
- **Parking Management**: Automating entry/exit of vehicles.
- **Traffic Analysis**: Congestion and journey time analysis.
- **Toll Collection**: Seamless toll payment automation.

---

## Technologies Used 🛠️

- **Programming Language**: Python
- **Libraries**:
  - TensorFlow
  - PyTorch
  - OpenCV
  - Numpy, Pandas, Matplotlib
- **Tools**:
  - TensorFlow Object Detection API
  - EasyOCR
  - Jupyter Notebook
  - Anaconda
- **Hardware**:
  - Windows 10 (64-bit)
  - AMD Ryzen 5 Gen 3
  - GPU: GeForce GTX 1650 4GB
  - 8GB RAM (minimum)

---

## Setup and Installation 🛠️

Follow these steps to set up the project on your local system:

### 1. Set Up TensorFlow Object Detection API:
Follow the TensorFlow Object Detection API setup [guide](https://tensorflow-object-detection-api-tutorial).

### 2. Dataset Preparation:
- Download the **CCPD** dataset:
  [CCPD Dataset Link](https://drive.google.com/open?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc)
- Place the dataset in the `data/` directory.

### 3. Run Training:
To train the TensorFlow model:

Detailed code are inside the project report

```bash
python train.py --model_dir=models/ssd_mobnet --pipeline_config_path=configs/pipeline.config
```

For the RPNet model:
```bash
python rpnet_train.py --images=data/CCPD --epochs=25
```

---

## Dataset 📊

We used the **CCPD (Chinese City Parking Dataset)**, which contains over **250k unique images** of vehicles. It provides accurate bounding box annotations for license plate localization.

---

## Training and Testing 🏋️‍♀️

### Model 1: TensorFlow Object Detection
- Pre-trained **SSD MobileNet** model fine-tuned on the CCPD dataset.
- Includes OCR filtering using **EasyOCR**.

### Model 2: RPNet
- Custom CNN-based model for simultaneous detection and recognition.

---

## Results 📈

- **TensorFlow Model**: Lightweight and fast; suitable for congestion analysis.
- **RPNet Model**: High accuracy for law enforcement and vehicle identification.

![image](https://github.com/user-attachments/assets/7ff94968-c145-4d7b-82a5-88618acbb528)


---

## Future Work 🚀

- Add support for multiple languages and license plate formats.
- Optimize models for edge devices like Raspberry Pi.
- Enhance accuracy using hybrid deep learning approaches.

---

## Contributors 🤝

- **Nishikanta Parida**  
  *Regd. No.: 2124100015*

**Supervisor**:  
- Dr. Debasis Gountia (Associate Professor, CSA Department)

---

### If you find this project useful, consider giving it a ⭐!
