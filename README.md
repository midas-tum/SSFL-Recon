# SSFL-Recon

**Self-supervised feature learning for cardiac Cine MR image reconstruction**

> Official Tensorflow implementation of our paper.

✏️ **Authors**: Siying Xu, Marcel Früh, Kerstin Hammernik, Andreas Lingg, Jens Kübler, Patrick Krumm, Daniel Rueckert, Sergios Gatidis, and Thomas Küstner  

---

## 🔧 Overview

SSFL-Recon is a two-stage framework for MR image reconstruction using self-supervised feature learning. It consists of:

1. **Self-supervised Feature Learning**: Learns sampling-insensitive global features from undersampled images using:
   - Contrastive Learning
   - VICReg (Variance-Invariance-Covariance Regularization)

2. **Self-supervised Reconstruction**: Embeds the learned features into a self-supervised reconstruction network.

---

## 📁 Project Structure

```bash
SSFL-Recon/
├── data_loader/                              # Data loading modules
│   ├── feature_contrastive_data.py           # Data pipeline for contrastive feature learning
│   ├── feature_contrastive_data_dummy.py  
│   ├── feature_vicreg_data.py                # Data pipeline for VICReg feature learning
│   ├── feature_vicreg_data_dummy.py          
│   ├── recon_data.py                         # Data pipeling for reconstruction
│   └── recon_data_dummy.py
│
├── evaluation/
│   ├── metrics.py                            # NRMSE, PSNR, SSIM
│
├── feature_learning/                         # Feature learning step
│   ├── losses/                               # Contrastive/VICReg loss functions
│   │   ├── contrastive_loss.py
│   │   └── vicreg_loss.py
│   ├── models/                               
│   │   ├── VICReg_feature_model.py           # model for VICReg feature learning
│   │   ├── base_unet.py                      # base complex 2D+t UNet
│   │   ├── contrastive_feature_model.py      # model for contrastive feature learning
│   │   ├── decoder_unet.py                  
│   │   ├── encoder_unet.py
│   │   └── mlp.py                            # multi-layer perceptron
│   └── train/                                # Training scripts (feature learning step)
│       ├── train_contrastive.py                                 
│       ├── train_contrastive_dummy.py
│       ├── train_vicreg.py
│       └── train_vicreg_dummy.py
│
├── reconstruction/                           # Feature-assisted self-supervised reconstruction
│   ├── dummy_weights/                        # Pretrained feature extractor (FE-Net) weights (dummy)
│   │   ├── weights001.tf.data-00000-of-00001
│   │   └── weights001.tf.index
│   ├── recon_model.py           
│   ├── feature_assisted_unet.py  
│   ├── recon_loss.py             
│   ├── train_SSFL_recon.py                   # Main training script
│   └── train_SSFL_recon_dummy.py             # Dummy training script
│
├── utils/                                    # Utility functions
│   ├── basic_functions.py
│   ├── callbacks.py
│   ├── data_consistency.py
│   ├── layers.py
│   └── mri.py
│
└── README.md
```

---

## 🧪 Dummy Test (No real dataset required)
You can run the entire feature learning + reconstruction pipeline with dummy data and dummy pre-trained weights:

```bash
# Run contrastive feature learning on dummy data
python feature_learning/train/train_contrastive_dummy.py

# Run reconstruction using dummy pre-trained features
python reconstruction/train_SSFL_recon_dummy.py
```


## 📂 Training with Real Dataset

This project uses an **in-vivo cardiac Cine MR dataset**, which cannot be publicly released due to institutional data sharing restrictions.

If you are interested in reproducing the results, please contact the authors for potential collaboration or use publicly available alternatives such as:

- [OCMR](https://www.ocmr.info/)
- [CMRxRecon](https://www.synapse.org/Synapse:syn51471091/wiki/622170)

---

## 📽️ Presentation

We presented SSFL-Recon with **contrastive feature learning** at the 2023 ISMRM & ISMRT Annual Meeting & Exhibition, and SSFL-Recon with **VICReg feature learning** at the 2024 ISMRM & ISMRT Annual Meeting & Exhibition.

🎞️ [▶ Watch the presentation of SSFL-Recon(c)](https://archive.ismrm.org/2023/0709.html)
🎞️ [▶ Watch the presentation of SSFL-Recon(v)](https://archive.ismrm.org/2024/0012.html)

> 📝 Both presentations were based on an earlier version of our work, submitted as abstracts to ISMRM 2023 & 2024.  
> The current repository reflects the full version of our paper.

---

## 📚 Citation

Coming soon.

---

## 📬 Contact

For questions or collaboration opportunities, feel free to reach out to Siying Xu at siying.xu@med.uni-tuebingen.de

