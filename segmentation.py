import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_features(image_path):
    features = {}

    # Load and convert image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Use HSV to segment white background
    lower_white = np.array([0, 0, 160])
    upper_white = np.array([180, 80, 255])
    mask_bg = cv2.inRange(img_hsv, lower_white, upper_white)
    mask = cv2.bitwise_not(mask_bg)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, filename), mask)  # Save mask for debugging

    # HSV statistics
    masked_hsv = img_hsv[mask == 255]
    masked_v = masked_hsv[:, 2]
    features["h_mean"] = np.mean(masked_hsv[:, 0])
    features["s_mean"] = np.mean(masked_hsv[:, 1])
    features["v_mean"] = np.mean(masked_hsv[:, 2])
    features["h_std"] = np.std(masked_hsv[:, 0])
    features["s_std"] = np.std(masked_hsv[:, 1])
    features["v_std"] = np.std(masked_hsv[:, 2])

    # Hue Histogram
    hist = cv2.calcHist([img_hsv], [0], mask, [16], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    for i in range(len(hist)):
        features[f"hist_h_{i}"] = hist[i]

    # Shape features
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        features["aspect_ratio"] = float(w) / h
        features["extent"] = float(area) / (w * h)
        features["solidity"] = float(area) / hull_area if hull_area != 0 else 0
        features["roundness"] = (
            (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0
        )

        # Elongation
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            features["elongation"] = major_axis / minor_axis

        # Hu Moments
        moments = cv2.moments(cnt)
        hu_moments = cv2.HuMoments(moments).flatten()
        for i in range(3):
            features[f"hu_{i}"] = -np.sign(hu_moments[i]) * np.log10(
                abs(hu_moments[i]) + 1e-10
            )

    # Texture features (GLCM)
    img_gray_norm = (img_gray / 255.0 * 15).astype(np.uint8)
    glcm = graycomatrix(img_gray_norm, [1], [0], 16, symmetric=True, normed=True)
    features["glcm_contrast"] = graycoprops(glcm, "contrast")[0, 0]
    features["glcm_homogeneity"] = graycoprops(glcm, "homogeneity")[0, 0]
    features["glcm_energy"] = graycoprops(glcm, "energy")[0, 0]
    features["glcm_entropy"] = -np.sum(glcm * np.log2(glcm + 1e-10))

    # Edge density
    edges = cv2.Canny(img_gray, 100, 200)
    features["edge_density"] = np.sum(edges) / (img_gray.shape[0] * img_gray.shape[1])

    # Symmetry (horizontal & vertical)
    h, w = img_gray.shape
    left = img_gray[:, : w // 2]
    right = img_gray[:, w // 2 :]
    right_flipped = np.fliplr(right)
    min_width = min(left.shape[1], right_flipped.shape[1])
    features["symmetry_horizontal"] = np.mean(
        np.abs(left[:, :min_width] - right_flipped[:, :min_width])
    )

    top = img_gray[: h // 2, :]
    bottom = img_gray[h // 2 :, :]
    bottom_flipped = np.flipud(bottom)
    min_height = min(top.shape[0], bottom_flipped.shape[0])
    features["symmetry_vertical"] = np.mean(
        np.abs(top[:min_height, :] - bottom_flipped[:min_height, :])
    )

    # Bright spot ratio
    features["bright_spot_ratio"] = np.mean(masked_v > 240)

    # Surface roughness (Laplacian)
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    features["laplacian_var"] = lap.var()

    return features
