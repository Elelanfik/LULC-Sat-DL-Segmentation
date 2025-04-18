# Enhancing Land Use and Land Cover Classification with Deep Learning-Based Satellite Imagery Segmentation

## Overview
This repository contains the code, data, and resources for the research paper titled **"Enhancing Land Use and Land Cover Classification with Deep Learning-Based Satellite Imagery Segmentation."** The study focuses on leveraging advanced deep learning architectures for semantic segmentation of satellite imagery to improve land use and land cover (LULC) classification. The research evaluates multiple state-of-the-art models and preprocessing techniques to identify the most effective approach for classifying satellite images into eight distinct land cover classes.

## Key Objectives
- Evaluate and compare deep learning architectures for semantic segmentation of satellite imagery.
- Create a high-resolution dataset with corresponding masks for land cover classification.
- Implement preprocessing techniques, including normalization and data augmentation, to enhance model performance.
- Identify the most effective model architecture and backbone combination for LULC classification.
- Provide a reproducible pipeline for researchers and practitioners working on similar tasks.

## Dataset
A high-resolution satellite imagery dataset was curated for this study, including corresponding masks for eight land cover classes:
1. Built-up areas
2. Roads
3. Water bodies
4. Agricultural land
5. Shrubland
6. Forest
7. Grassland
8. Others

The dataset is split into training, validation, and testing sets to ensure robust evaluation and generalization across larger geographical areas.

## Preprocessing
The following preprocessing steps were applied to the dataset:
- **Normalization**: Standardized pixel values to improve model convergence.
- **Data Augmentation**: Techniques such as vertical and horizontal flipping, as well as random brightness adjustments, were used to increase dataset variability and reduce overfitting.

## Models Evaluated
Several advanced deep learning architectures were evaluated, including:
- **UNet**
- **LinkNet**
- **DeepLabV3+**
- **AE-DeepLabV3+**

These models were tested with various backbones, such as ResNet101, ResNet152, Xception, and MobileNetV2, to identify the best-performing combination.

## Results
The study demonstrated that certain model architectures and backbone combinations achieved superior performance in land use and land cover classification. Among the evaluated models, **AE-DeepLabV3+** with the **Xception backbone** emerged as the top-performing model, showcasing its effectiveness in this domain. The results were compared with recent studies to provide context and highlight the contributions of this research.

## Repository Structure
```
LULC-DeepSeg/
├── data/                   # Dataset and masks
├── models/                # Model architectures and weights
├── scripts/               # Preprocessing and training scripts
├── results/               # Evaluation results and visualizations
├── README.md              # This file
└── requirements.txt       # Dependencies
```

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Elelanfik/LULC-DeepSeg.git
   cd LULC-DeepSeg
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the Data**:
   Run the preprocessing scripts to normalize and augment the dataset:
   ```bash
   python scripts/preprocess.py
   ```

4. **Train the Models**:
   Train the models using the provided scripts:
   ```bash
   python scripts/train.py --model unet --backbone xception
   ```

5. **Evaluate the Models**:
   Evaluate the trained models on the test set:
   ```bash
   python scripts/evaluate.py --model aedeeplabv3plus --backbone xception
   ```

## Contribution
This repository is intended to serve as a resource for researchers and practitioners working on land use and land cover classification. Contributions, suggestions, and feedback are welcome! Please feel free to open an issue or submit a pull request.

## Citation
If you find this repository useful for your research, please consider citing our work:
```bibtex

```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

For any questions or inquiries, please contact [Your Email Address].
