# SSFL-Recon

**Self-supervised Feature Learning for Cardiac Cine MR Image Reconstruction**

> Official Tensorflow implementation of our paper.

✏️ **Authors**: Siying Xu, Marcel Früh, Kerstin Hammernik, Andreas Lingg, Jens Kübler, Patrick Krumm, Daniel Rueckert, Sergios Gatidis, and Thomas Küstner  

## 🔧 Overview

SSFL-Recon is a two-stage framework for MR image reconstruction using self-supervised feature learning. It consists of:

1. **Feature Learning Stage**: Learns sampling-insensitive global features from undersampled images using:
   - Contrastive Learning
   - VICReg (Variance-Invariance-Covariance Regularization)

2. **Self-supervised Reconstruction Stage**: Embeds the learned features into a self-supervised reconstruction network.

## 📁 Project Structure

