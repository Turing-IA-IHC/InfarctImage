# InfarctImage - Fine-Tuned Model for Heart Attack Simulation

![sd-2 1-infarct-lora-051-3501614513](https://github.com/user-attachments/assets/b22642fb-4952-497d-aad7-4abf8d2bcdb7)

## 📌 Overview
**InfarctImage** is a LoRA-based model fine-tuned on **Stable Diffusion 2.1** to generate realistic images of individuals simulating a heart attack. This model was developed to facilitate synthetic dataset generation for human activity recognition and medical emergency monitoring applications.

🔗 **Related Resources:**
- **Article:** [Expanding Domain-Specific Datasets with Generative Models for Simulating Heart Attacks](#)
- **Kaggle Dataset:** [InfarctImage Dataset](https://www.kaggle.com/datasets/gavit0/InfarctImage)

## 🚀 Getting Started

### Prerequisites
To use this model, install the required dependencies:
```bash
pip install diffusers torch torchvision kaggle
```

### Installation
Like a library from repository
```bash
pip install git+https://github.com/Turing-IA-IHC/InfarctImage.git
```
Or clone this repository and navigate to its directory:
```bash
git clone https://github.com/Turing-IA-IHC/InfarctImage.git
cd InfarctImage
```

## 🎯 Usage
Load and use the model with the following Python script:
```python
# Import the custom library
from infarctimage import InfarctImageCreator
# Initialize the infarct model
infarct_model = InfarctImageCreator()
# Generate images
prompt = ("Elderly man at a sports stadium surrounded by a crowd, "
          "clutching his chest with a distressed look, indicating a heart attack."
         )
image = infarct_model.generate_image(prompt)
image.show()
```

Full examples in: 
- 🔗 **[Notebook in GitHub](notebooks/infarctImageDemo.ipynb)**
- [![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Turing-IA-IHC/InfarctImage/blob/main/notebooks/infarctImageDemo.ipynb)

## 📊 Model Training
This model was trained on a dataset of 100 manually labeled images:
- 50 images of individuals simulating heart attack symptoms.
- 50 images of individuals in neutral contexts.

Annotations were enhanced using **BLIP (Bootstrapping Language-Image Pretraining)** to improve descriptive quality.

### Training Configuration
- **Base Model:** Stable Diffusion 2.1
- **Fine-Tuning Technique:** LoRA (Low-Rank Adaptation)
- **Learning Rate:** 0.0001
- **Batch Size:** 1
- **Epochs:** 10
- **Hardware:** NVIDIA RTX 4090 (24GB VRAM)

## 📈 Evaluation Metrics
**LPIPS (Learned Perceptual Image Patch Similarity)** was used to assess image quality:

| Model | LPIPS (↓ Better) |
|--------|---------------|
| SD 2.1 Base | 0.7366 |
| SD 2.1 + LoRA | 0.6919 |

## 🤝 Contributing
We welcome contributions! To contribute:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Added new feature'`)
4. Push to your branch (`git push origin feature-branch`)
5. Open a Pull Request

## 📜 License
This project is licensed under the **MIT License**.

## 📧 Contact
For questions or collaborations, contact **lgabrielrojas@ucundinamarca.edu.co** or open an issue in this repository.

---
