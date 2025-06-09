import os
import pandas as pd
from segmentation import extract_features


def generate_feature_dataframe(image_dir):
    """
    Some simple example code for using the extract_features function
    you can modify this to suit your needs
    """
    df = pd.DataFrame()

    for fname in os.listdir(image_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(image_dir, fname)
            row = extract_features(path)
            row_df = pd.DataFrame([row])
            row_df["filename"] = fname
            df = pd.concat([df, row_df], ignore_index=True)

    return df


if __name__ == "__main__":
    """image_dir = "Dataset/Classification_dataset/Chaunsa (White)"
    output_csv = "mango_features.csv"  # 輸出 CSV
    df = generate_feature_dataframe(image_dir)
    df.to_csv(output_csv, index=False)
    print(f"Features extracted and saved to {output_csv}")
    """
    path = "Dataset/Classification_dataset/Chaunsa (White)/IMG_20210705_101052.jpg"
    features = extract_features(path)
    for key, value in features.items():
        print(f"{key}: {value}")
