# 🧠 Gen AI IITM Tutorials

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Overview

Welcome to **Gen AI IITM Tutorials** – a comprehensive collection of hands-on deep learning notebooks focused on **Generative Adversarial Networks (GANs)** and **Variational Autoencoders (VAEs)**. This repository is designed for learners and practitioners who want to master generative modeling using PyTorch, with a special emphasis on clarity, reproducibility, and best engineering practices.

This repository covers both major paradigms of generative AI:
- **GANs**: Adversarial training for high-quality image generation
- **VAEs**: Probabilistic latent space modeling for controlled generation

---

## ✨ Features

### 🤖 GAN Architectures
- **Vanilla GAN** - Basic GAN implementation
- **Deep Convolutional GAN (DCGAN)** - Convolutional GAN for image generation
- **Conditional GAN (cGAN)** - Class-conditional image generation
- **Wasserstein GAN (WGAN)** - Improved training stability
- **BiGAN** - Bidirectional GAN for representation learning

### 🔮 VAE Architectures
- **Beta-VAE** - Disentangled representation learning with controllable β parameter
- **Vector Quantized VAE (VQ-VAE)** - Discrete latent representations

### 🛠️ Technical Features
- **Clean, Modular PyTorch Code**
- **Well-Documented Jupyter Notebooks**
- **MNIST Dataset Integration**
- **Image Generation & Visualization**
- **Easy to Extend for Custom Datasets**
- **Production-Ready Engineering Practices**
- **Step-by-Step Explanations & Comments**
- **GPU Memory Management**
- **Comprehensive Evaluation Metrics**

---

## 📂 Repository Structure

```
Gen-AI-IITM-Tutorials/
│
├── GAN_Architectures/           # GAN implementations
│   ├── Vanilla_GAN.ipynb       # Basic GAN
│   ├── DC_GAN.ipynb            # Deep Convolutional GAN
│   ├── Conditional_GAN.ipynb   # Conditional GAN
│   ├── WGAN.ipynb              # Wasserstein GAN
│   ├── Bi_GAN.ipynb            # Bidirectional GAN
│   ├── generated_images/       # GAN output samples
│   ├── DC_GAN_generated_images/
│   ├── cgan_generated/
│   ├── w_gan_dcgan/
│   └── data/                   # Dataset storage
│
├── VAE_Architectures/          # VAE implementations
│   ├── Beta_VAE.ipynb         # Beta-VAE with disentanglement
│   ├── VQ_VAE.ipynb           # Vector Quantized VAE
│   └── data/                  # Dataset storage
│
├── Assignments/               # Course assignments
│   └── OPPE_Mock_WGAN_250816_123209.pdf
│
├── .ipynb_checkpoints/       # Jupyter notebook checkpoints
└── README.md                 # This file
```

---

## 🛠️ Setup & Usage

1. **Clone the repository**
    ```sh
    git clone https://github.com/AnirudhNarayan/Gen-AI-Tutorials.git
    cd Gen-AI-Tutorials
    ```

2. **Install dependencies**
    ```sh
    pip install torch torchvision matplotlib notebook numpy
    ```

3. **Launch Jupyter Notebook**
    ```sh
    jupyter notebook
    ```
    Navigate to either `GAN_Architectures/` or `VAE_Architectures/` and open any notebook!

---

## 🎯 Learning Path

### For Beginners
1. Start with `GAN_Architectures/Vanilla_GAN.ipynb`
2. Progress to `GAN_Architectures/DC_GAN.ipynb`
3. Explore `VAE_Architectures/Beta_VAE.ipynb`

### For Intermediate Users
1. Try `GAN_Architectures/Conditional_GAN.ipynb`
2. Experiment with `GAN_Architectures/WGAN.ipynb`
3. Dive into `VAE_Architectures/VQ_VAE.ipynb`

### For Advanced Users
1. Explore `GAN_Architectures/Bi_GAN.ipynb`
2. Customize architectures for your datasets
3. Experiment with different loss functions and hyperparameters

---

## 🖼️ Sample Results

### GAN Generations
| Vanilla GAN | DCGAN | Conditional GAN |
|:-----------:|:-----:|:--------------:|
| ![Vanilla GAN](GAN_Architectures/generated_images/vanilla_sample.png) | ![DCGAN](GAN_Architectures/generated_images/dcgan_sample.png) | ![cGAN](GAN_Architectures/generated_images/cgan_sample.png) |

### VAE Reconstructions
| Original | Beta-VAE Reconstruction | Latent Space Visualization |
|:--------:|:----------------------:|:---------------------------:|
| ![Original](VAE_Architectures/data/original_sample.png) | ![Beta-VAE](VAE_Architectures/data/reconstruction_sample.png) | ![Latent Space](VAE_Architectures/data/latent_space.png) |

---

## 📚 Learning Outcomes

### GANs
- Understand adversarial training dynamics
- Implement various GAN architectures from scratch
- Learn techniques for stable GAN training
- Master conditional generation

### VAEs
- Grasp probabilistic latent space modeling
- Implement variational inference
- Understand disentanglement in Beta-VAE
- Learn discrete representations with VQ-VAE

### General Skills
- **PyTorch Mastery**: Deep understanding of PyTorch for generative models
- **Research Skills**: Ability to implement and extend research papers
- **Engineering Best Practices**: Clean, modular, and production-ready code
- **Visualization**: Create compelling visualizations for generative models

---

## 🔬 Key Concepts Covered

### GAN Concepts
- **Adversarial Training**: Generator vs Discriminator dynamics
- **Mode Collapse**: Understanding and mitigation strategies
- **Conditional Generation**: Class-conditional image synthesis
- **Wasserstein Distance**: Improved training stability
- **Bidirectional Learning**: Joint optimization of generation and inference

### VAE Concepts
- **Variational Inference**: Approximate posterior inference
- **ELBO Optimization**: Evidence Lower BOund maximization
- **Disentanglement**: Learning interpretable latent factors
- **Discrete Representations**: Vector quantization for discrete latent spaces
- **β-Scheduling**: Controlling the trade-off between reconstruction and KL divergence

---

## 👨‍💻 Author

**Anirudh Narayan**  
*Deep Learning Enthusiast | IITM | Open Source Contributor*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/anirudhnarayan/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github)](https://github.com/AnirudhNarayan)

---

## 🏆 Why This Project Stands Out

- **Comprehensive Coverage**: Both GANs and VAEs in one repository
- **Clean, Industry-Level Code**: Modular design with best practices
- **Rich Documentation**: In-line explanations and step-by-step tutorials
- **Research-Ready**: Easy to extend for new architectures and datasets
- **Production-Ready**: GPU memory management and error handling
- **Impressive Portfolio**: Showcases advanced generative AI skills

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- IITM for providing the learning platform
- PyTorch team for the excellent deep learning framework
- The generative AI research community for continuous innovations

---

> *"Generative AI is the future. Start building it today!"* 🚀

---

**⭐ Star this repository if you found it helpful!**