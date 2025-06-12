import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def run_classification(csv_file):
    # 1. 讀取資料
    df = pd.read_csv(csv_file)

    # 2. 分離特徵與標籤
    X = df.drop(columns=["filename", "label"])
    y = df["label"]

    # 3. 特徵標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 訓練/測試分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=123
    )

    # 5. 訓練 MLP 模型
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=123)
    clf.fit(X_train, y_train)

    # 6. 預測與報告
    y_pred = clf.predict(X_test)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. 混淆矩陣
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clf.classes_,
        yticklabels=clf.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - MLP")
    plt.tight_layout()
    plt.show()
