# Image Processing final project
This project extracts a variety of features from mango images for classification or analysis. It uses OpenCV, NumPy, scikit-image, and pandas to process images and output a CSV file of features.

## Features
- HSV color statistics
- Hue histogram
- Shape descriptors (aspect ratio, extent, solidity, roundness, elongation, Hu moments)
- Texture features (GLCM)
- Edge density
- Symmetry (horizontal & vertical)
- Bright spot ratio
- Surface roughness

## Installation
Clone the repository and install dependencies and copy the dataset to `dataset` directory:
```sh
git clone git@github.com:Darren1225/ImageProcessingFinal.git
cd ImageProcessingFinal
pip install -r requirements.txt
```

## Usage
Modify the path in `main.py` and run this function!
```sh
python main
```
A CSV file containing the extracted features for each image.