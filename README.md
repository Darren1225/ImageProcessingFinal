# Image Processing Final Project

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

Clone the repository, install dependencies, and copy the dataset to the `Dataset` directory:

```sh
git clone git@github.com:Darren1225/ImageProcessingFinal.git
cd ImageProcessingFinal
pip install -r requirements.txt
```

## Usage

Modify the `image_dir` path in `main` if needed, then run:

```sh
python main
```

A CSV file containing the extracted features for each image will be generated.

## Development Workflow

Please follow these development rules:

1. **Create a New Branch**
   - For any new feature or bug fix, create a new branch from `main`.
   - Use descriptive branch names, such as `feature/add-glcm-feature` or `fix/bug-mask-shape`.
   - Command example:
     ```sh
     git switch main
     git pull
     git switch -c feature/your-feature-name
     ```

2. **Develop on Your Branch**
   - Work and test your code on your own branch. Do not commit directly to `main`.
   - Make frequent commits with clear commit messages.
   - Example:
     ```sh
     git add your_changed_files.py
     git commit -m "Clear and descriptive commit message"
     git push origin feature/your-feature-name
     ```

3. **Open a Pull Request (PR)**
   - After finishing your work, push your branch to GitHub and open a Pull Request to `main`.
   - Clearly describe your changes and purpose in the PR description.

4. **Code Review**
   - The team leader will review every PR for code quality and correctness.
   - If changes are requested, address the feedback and update your PR.

5. **Merge**
   - Only the team leader merges PRs into `main` after approval.
   - Do not merge your own PRs.

6. **Additional Notes**
   - Keep code style consistent and add comments where necessary.
   - Update `requirements.txt` if you add new dependencies.
   - For questions or suggestions, use GitHub Issues or group chat.

If you notice any missing or unclear rules, please bring them up so we can improve our workflow together!