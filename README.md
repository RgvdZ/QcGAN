# 🚀 QcGAN: Fast Motion Deblurring using Conditional GANs

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

> A lightweight, blazing-fast PyTorch implementation of **Quick deblurring using cGAN** (QcGAN).

This repository contains the complete training and modeling pipeline to perform blind motion deblurring on single images. By leveraging MobileNet-inspired depthwise separable convolutions, this model achieves state-of-the-art structural similarity and visual appearance while radically reducing parameter count and inference time.

---

<img width="551" height="250" alt="image" src="https://github.com/user-attachments/assets/4e0c29e7-c7c4-4a87-bd99-973c693faecf" />


## ✨ Key Features

* **Lightweight Architecture**: Replaces standard convolutions with **Depthwise Separable Convolutions** in the ResNet blocks, reducing the model size by 3-60x compared to competitors.
* **PatchGAN Discriminator**: Utilizes a Markovian discriminator architecture to ensure high-frequency detail restoration and rich color synthesis.
* **Advanced Loss Functions**: Optimized using a combination of **Hinge Loss** (for stable adversarial training) and **VGG-16 Perceptual Loss** (for photo-realistic content structure).
* **Real-Time Ready**: Designed specifically with edge-devices and resource-constrained environments (like robotics and mobile) in mind.

---

<img width="551" height="250" alt="image" src="https://github.com/user-attachments/assets/bc8462c6-ec47-4e45-93f1-be4dca3f89b3" />


## 📂 Repository Structure

The project is modularized into four clean, highly readable files:

```text
├── dataset.py      # Custom PyTorch Dataset for paired Blur/Sharp image loading & augmentation
├── model.py        # Contains the Generator, Discriminator, and VGG Perceptual Loss networks
├── train.py        # The core training loop, dynamic learning rates, and checkpointing logic
├── main.py         # The entry point to initialize hyperparameters and execute training
└── README.md       # You are here!
