import pandas as pd
import matplotlib.pyplot as plt

def summarize_data(csv_path):
    df = pd.read_csv(csv_path)

    print("\n=== Basic Info ===")
    print(df.info())

    print("\n=== Descriptive Stats ===")
    print(df.describe())

    # Correlation heatmap
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr)

    plt.title("Correlation Heatmap of Audio Features", fontsize=16)

    # Add tick labels
    features = corr.columns
    plt.xticks(range(len(features)), features, rotation=45, ha="right")
    plt.yticks(range(len(features)), features)

    # Colorbar
    plt.colorbar(label="Correlation Strength")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    summarize_data("liked_songs.csv")
