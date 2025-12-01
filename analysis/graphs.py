import pandas as pd
import matplotlib.pyplot as plt

FEATURES = [
    "danceability", "energy", "valence",
    "acousticness", "speechiness", "instrumentalness",
    "liveness", "tempo"
]


def plot_histograms(df, title):
    df[FEATURES].hist(bins=20, figsize=(12, 10))
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def scatter_feature(df, x, y):
    plt.scatter(df[x], df[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs. {y}")
    plt.show()


if __name__ == "__main__":
    liked = pd.read_csv("liked_songs.csv")
    plot_histograms(liked, "Liked Songs Feature Distribution")
    # scatter_feature(liked, "energy", "valence")
