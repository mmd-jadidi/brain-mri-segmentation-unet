# Brain MRI Tumor Segmentation using U-Net

An end-to-end deep learning project for automatic segmentation of Lower-Grade Glioma (LGG) abnormalities in Brain MRI scans using a **U-Net** convolutional neural network with custom hybrid loss and dynamic Otsu post-processing.

## Project Overview

Identifying and delineating brain tumors from magnetic resonance imaging (MRI) is essential for diagnosis, surgical planning, and radiotherapy. This project implements:
1. **Dataset Pipeline**: Preprocessing FLAIR, pre-contrast, and post-contrast MRI slices with corresponding binary abnormality masks.
2. **U-Net Architecture**: An encoder-decoder deep neural network with skip connections for high-resolution feature preservation.
3. **Hybrid Loss Function**: Combined Binary Crossentropy and Dice Loss to tackle extreme foreground-background class imbalance.
4. **Post-Processing Pipeline**: Dynamic intensity thresholding via Otsu's method and morphological filtering (removal of false-positive small isolated connected components).
5. **Evaluation**: Quantitative metrics including Intersection-over-Union (IoU / Jaccard Index), Dice coefficient, and Pixel Accuracy.

## Repository Structure

```text
├── brain_mri_segmentation.ipynb       # Complete end-to-end training, evaluation & visualizations
├── test.py                            # Standalone script for sample inference & evaluation
├── brain_mri_segmentation_report.pdf  # Complete project documentation and analytical report
├── brain_mri_segmentation_report.html # Exported HTML notebook report
└── README.md                          # Project overview and documentation
```

## Dataset

This project uses the **LGG Segmentation Dataset** from The Cancer Genome Atlas (TCGA) and The Cancer Imaging Archive (TCIA):
- **Samples**: 3,929 FLAIR abnormality MRI slices across 110 patients with lower-grade gliomas.
- **Image Format**: 3-channel `.tif` sequences (pre-contrast, FLAIR, post-contrast) resized to 256x256.
- **Masks**: 1-channel binary ground-truth segmentation masks.
- **Download**: Available on [Kaggle: LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation).

Place the dataset inside a `kaggle_3m/` directory in the repository root or update the dataset path in the notebook.

## Architecture

The model uses a symmetric U-Net architecture:
- **Contracting Path (Encoder)**: Consecutive $3\times3$ Conv2D blocks with ReLU activation, followed by $2\times2$ MaxPooling for multi-scale feature extraction.
- **Bottleneck**: Deep convolutional representation of latent features.
- **Expanding Path (Decoder)**: $2\times2$ Conv2DTranspose upsampling layers concatenated with corresponding encoder feature maps via skip connections.
- **Output Layer**: $1\times1$ Conv2D with Sigmoid activation producing a per-pixel probability map.

## Installation & Requirements

Ensure Python 3.9+ and install required dependencies:

```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn scikit-image
```

## Running the Project

1. **Jupyter Notebook**:
   ```bash
   jupyter notebook brain_mri_segmentation.ipynb
   ```
   Execute the cells to run data loading, model compilation, training, and result visualization.

2. **Inference Test Script**:
   ```bash
   python3 test.py
   ```

## Documentation & Report

For in-depth analysis, loss curves, ablation studies, and qualitative results, see the attached documentation:
- [`brain_mri_segmentation_report.pdf`](brain_mri_segmentation_report.pdf)
- [`brain_mri_segmentation_report.html`](brain_mri_segmentation_report.html)
