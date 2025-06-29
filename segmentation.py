import cv2
import os
import numpy as np


def edge_density(img_gray):
    # Step 1: Gaussian Blur
    blur = cv2.GaussianBlur(img_gray, (5, 5), 1.4)  # not sure if allowed

    # Step 2: Gradients
    Gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)  # not sure if allowed
    Gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    angle[angle < 0] += 180

    # Step 3: Non-Maximum Suppression
    nms = np.zeros_like(magnitude, dtype=np.uint8)
    rows, cols = magnitude.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = angle[i, j]
            before = after = 0

            # (i-1, j-1)   (i-1, j)   (i-1, j+1)
            # (i,   j-1)   (i,   j)   (i,   j+1)
            # (i+1, j-1)   (i+1, j)   (i+1, j+1)
            # Horizontal
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                before = magnitude[i, j - 1]
                after = magnitude[i, j + 1]
            # ↘
            elif 22.5 <= direction < 67.5:
                before = magnitude[i - 1, j + 1]
                after = magnitude[i + 1, j - 1]
            # Vertical
            elif 67.5 <= direction < 112.5:
                before = magnitude[i - 1, j]
                after = magnitude[i + 1, j]
            # ↙
            elif 112.5 <= direction < 157.5:
                before = magnitude[i - 1, j - 1]
                after = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= before and magnitude[i, j] >= after:
                nms[i, j] = magnitude[i, j]

    # Step 4: Double Threshold (Customizable))
    high_threshold = 100  # choose a value above which is considered a strong edge
    low_threshold = 50  # choose a value below which is considered a weak edge

    strong = 255
    weak = 75
    res = np.zeros_like(nms, dtype=np.uint8)

    strong_i, strong_j = np.where(nms >= high_threshold)
    weak_i, weak_j = np.where((nms >= low_threshold) & (nms < high_threshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    # Step 5: Hysteresis (eight-connectivity)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if res[i, j] == weak:
                if (
                    (res[i + 1, j - 1] == strong)
                    or (res[i + 1, j] == strong)
                    or (res[i + 1, j + 1] == strong)
                    or (res[i, j - 1] == strong)
                    or (res[i, j + 1] == strong)
                    or (res[i - 1, j - 1] == strong)
                    or (res[i - 1, j] == strong)
                    or (res[i - 1, j + 1] == strong)
                ):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    # Step 6: Compute edge density (strong edges only)
    edge_density = np.sum(res == strong) / (rows * cols)

    return {"edge_density": edge_density}


def GLCM(
    img_gray, levels=16, directions=[(1, 0), (0, 1), (1, 1), (-1, 1)]
):  # levels=16 -> disctretize the image into 16 gray levels
    img_norm = ((img_gray / 255.0) * (levels - 1)).astype(np.uint8)
    h, w = img_norm.shape

    # Initialize accumulators for features
    total_glcm = np.zeros((levels, levels), dtype=np.float64)
    contrast_total = 0
    homogeneity_total = 0
    energy_total = 0
    entropy_total = 0

    for dx, dy in directions:
        glcm = np.zeros((levels, levels), dtype=np.float64)

        # Loop through pixels, avoiding out-of-bounds access
        for y in range(h - abs(dy)):
            for x in range(w - abs(dx)):
                y1, x1 = y, x
                y2, x2 = y + dy, x + dx

                # Ensure target pixel stays within bounds
                if 0 <= x2 < w and 0 <= y2 < h:
                    i = img_norm[y1, x1]
                    j = img_norm[y2, x2]
                    glcm[i, j] += 1
                    glcm[j, i] += 1  # Symmetric

        glcm /= glcm.sum() + 1e-10  # Normalize to avoid division by zero

        # Feature calculations
        i_indices, j_indices = np.indices(glcm.shape)
        contrast = np.sum(glcm * (i_indices - j_indices) ** 2)
        homogeneity = np.sum(glcm / (1.0 + np.abs(i_indices - j_indices)))
        energy = np.sqrt(np.sum(glcm**2))  ## L2 norm(sqare root of sum of squares
        entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

        # Accumulate features
        contrast_total += contrast
        homogeneity_total += homogeneity
        energy_total += energy
        entropy_total += entropy

        total_glcm += glcm

    n = len(directions)
    return {
        "glcm_contrast": contrast_total / n,
        "glcm_homogeneity": homogeneity_total / n,
        "glcm_energy": energy_total / n,
        "glcm_entropy": entropy_total / n,
    }


def Symmetry(img_gray):

    # Ensure the image is grayscale (height x width)
    if len(img_gray.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    # Horizontal symmetry
    h, w = img_gray.shape
    left = img_gray[:, : w // 2]
    right = img_gray[:, w // 2 :]
    right_flipped = np.fliplr(right)  # flip left to right
    min_width = min(left.shape[1], right_flipped.shape[1])
    symmetry_horizontal = np.mean(
        np.abs(left[:, :min_width] - right_flipped[:, :min_width])
    )

    # Vertical symmetry
    top = img_gray[: h // 2, :]
    bottom = img_gray[h // 2 :, :]
    bottom_flipped = np.flipud(bottom)  # flip top to bottom
    min_height = min(top.shape[0], bottom_flipped.shape[0])
    symmetry_vertical = np.mean(
        np.abs(top[:min_height, :] - bottom_flipped[:min_height, :])
    )

    return {
        "symmetry_horizontal": symmetry_horizontal,
        "symmetry_vertical": symmetry_vertical,
    }


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
    features.update(
        GLCM(img_gray, levels=16, directions=[(1, 0), (0, 1), (1, 1), (-1, 1)])
    )

    # Edge density
    features.update(edge_density(img_gray))  # Edge Density features

    # Symmetry (horizontal & vertical)
    features.update(Symmetry(img_gray))

    # Bright spot ratio
    features["bright_spot_ratio"] = np.mean(masked_v > 240)

    # Surface roughness (Laplacian)
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    features["laplacian_var"] = lap.var()

    return features
