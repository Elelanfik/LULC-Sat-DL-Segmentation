
# Enhancing Land Use and Land Cover Classification with Deep Learning-Based Satellite Imagery Segmentation – **Part 2**

## 📘 Overview

This repository represents **Part 2** of the project titled:  
**"Enhancing Land Use and Land Cover Classification with Deep Learning-Based Satellite Imagery Segmentation."**

While **[Part 1](https://github.com/Elelanfik/LULC-Sat-DL-Segmentation)** contains the baseline model implementations all UNet, LinkNet,DeepLabV3+ and AE-DeepLabV3+ and baseline model implementations (UNet, LinkNet), **this repository focuses exclusively on advanced model training, evaluation, and results**, specifically for **DeepLabV3+** and **AE-DeepLabV3+** architectures.

> 📁 **Part 1 Repository:** [LULC-Sat-DL-Segmentation](https://github.com/Elelanfik/LULC-Sat-DL-Segmentation)

---

## 🎯 Key Objectives

- Train and evaluate **DeepLabV3+** and **AE-DeepLabV3+** on high-resolution satellite imagery
- Analyze performance across various backbones (e.g., ResNet101, ResNet152, Xception)
- Provide visual and metric-based evaluation for LULC classification tasks
- Serve as an extendable and reproducible framework for semantic segmentation research

---

## 📦 Dataset

The dataset used in this repository is the same as in Part 1. It contains satellite images and masks for eight land cover classes:
1. Built-up areas  
2. Roads  
3. Water bodies  
4. Agricultural land  
5. Shrubland  
6. Forest  
7. Grassland  
8. Others  

You can either clone or link to the dataset from Part 1.

---

## ⚙️ Preprocessing

The following preprocessing steps are reused from Part 1:
- **Normalization**: Scaling pixel values for better model convergence
- **Data Augmentation**: Flip, brightness change, etc., for improved generalization

---

## 🧠 Models Evaluated

This repository evaluates the following models:

- **DeepLabV3+**
- **AE-DeepLabV3+** (Atrous Encoder version)

Each tested with multiple backbones:
- ResNet101
- ResNet152
- Xception

---

## ✅ Results Summary

The study revealed that:
- **AE-DeepLabV3+ with the Xception backbone** performed best across all metrics
- Visual results showed improved segmentation in complex land types (e.g., shrub vs. forest)

Detailed evaluations and visualizations can be found in the `/results and Evaluation/` folder.

---

## 📁 Repository Structure

```
LULC-DeepSeg-Part2/
├── models/                      # DeepLabV3+ and AE-DeepLabV3+ architectures
├── results and Evaluation/     # Output metrics and visualizations
├── scripts/                    # Training and evaluation scripts
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## 🚀 Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/Elelanfik/LULC-DeepSeg-Part2](https://github.com/Elelanfik/LULC-Sat-DL-Segmentation_part2)
cd LULC-Sat-DL-Segmentation_part2
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Link or Prepare the Dataset

Use the dataset from Part 1 or your local path. Ensure proper folder structure under `data/`.

### 4. Train the Models

```bash
python scripts/train.py --model deeplabv3plus --backbone resnet101
```

Or for AE-DeepLabV3+:
```bash
python scripts/train.py --model aedeeplabv3plus --backbone xception
```

### 5. Evaluate the Models

```bash
python scripts/evaluate.py --model aedeeplabv3plus --backbone xception
```

---

## 🤝 Contribution

We welcome contributions, suggestions, and improvements!  
Feel free to open an issue or submit a pull request.

---

## 📚 Citation

If this project was helpful in your work, please consider citing:

```bibtex
```
---
## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.
---

## 📬 Contact

📧 fikadutsion@gmail.com  
📧 tsionhelalan@gmail.com
