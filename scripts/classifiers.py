import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans, Birch


def kmeans_classification(features, n_clusters=2):
    k_means = KMeans(n_clusters=n_clusters, n_init=1000, max_iter=10000)
    k_means.fit(features)
    labels = pd.Series(k_means.labels_, name="KMeans")
    # logging.debug(labels.to_string())
    # TS = labels.where(labels==1)
    # protein = labels.where(labels==0)
    return k_means, labels


def birch_classification(features, n_clusters=2):
    birch = Birch(n_clusters=n_clusters)
    birch.fit(features)
    labels = pd.Series(birch.labels_, name="Birch")
    return birch, labels


def plot_result_classif(regions, properties, labels, image):
    # TODO intégrer cette visulisation dans la Demo Region
    fig, axs = plt.subplots(ncols=2, figsize=(10, 6))
    # axs[0].imshow(image, cmap='gray')
    axs[0].imshow(image, cmap='gnuplot2')

    for i in range(len(regions)):
        # draw rectangle around segmented coins
        region = regions[i]
        minr, minc, maxr, maxc = region.bbox
        if labels[i] == 1:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)
        elif labels[i] == 0:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='blue', linewidth=1)
        else:
            raise ValueError(" Invalied with labels values")
        axs[0].add_patch(rect)

    sns.scatterplot(size=properties['extent'], hue=labels, x=properties['area'], y=properties["mean_intensity"],
                    ax=axs[1])
    axs[0].set_axis_off()
    plt.tight_layout()