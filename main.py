import os
import pandas as pd
from segmentation import extract_features
from classification import run_classification
import matplotlib.pyplot as plt


def generate_feature_dataframe(root_dir):
    """
    萃取所有子資料夾中的圖片特徵，並自動將子資料夾名稱作為 label。
    """
    df = pd.DataFrame()

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # 忽略不是資料夾的檔案

        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(class_path, fname)
                row = extract_features(path)
                row_df = pd.DataFrame([row])
                row_df["filename"] = fname
                row_df["label"] = class_name  # 用資料夾名稱作為分類標籤
                df = pd.concat([df, row_df], ignore_index=True)

    return df


if __name__ == "__main__":
    image_dir_0 = "Dataset/Classification_dataset"
    output_csv_0 = "classification_features.csv"
    image_dir_1 = "Dataset/Grading_dataset"
    output_csv_1 = "grading_features.csv"
    print("📥 正在從圖片萃取特徵...")
    df_0 = generate_feature_dataframe(image_dir_0)
    df_1 = generate_feature_dataframe(image_dir_1)
    df_0 = df_0.dropna()
    df_1 = df_1.dropna()
    df_0.to_csv(output_csv_0, index=False)
    print(f"✅ 特徵已儲存至 {output_csv_0}")
    df_1.to_csv(output_csv_1, index=False)
    print(f"✅ 特徵已儲存至 {output_csv_1}")
    print("\n🚀 開始訓練分類模型並顯示結果...")
    run_classification(output_csv_0)
    run_classification(output_csv_1)
