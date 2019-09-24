import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels
import seaborn as sns
import numpy as np

from sklearn.cluster import KMeans, Birch, MeanShift
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import scripts.file_manager as fm

#TODO :
# - Test KernelPCA
# - Test parameters of shiftMeans
from scripts.file_viewer import MultiLayerViewer


def classify(features):
    """ Complete Pipeline used classify features and identify transcription sites

    :param features:
    :return:
    """
    f = tail_filter(features)
    X = normalize_and_center(f)
    X = principal_component_analysis(X)
    labels = mean_shift_classification(X)
    labels.name="label"
    ts_label = labels.value_counts().idxmax()
    f_label = pd.concat([f.reset_index(), labels], axis=1).set_index('index')
    df_ts = f_label.loc[f_label["label"]==0]
    print(len(df_ts))
    return f_label


def mean_shift_classification(features):
    ms = MeanShift()
    ms.fit(features)
    labels = pd.Series(ms.labels_, name="mean_shift")
    return labels


def kmeans_classification(features, n_clusters=2, init='random'):
    k_means = KMeans(n_clusters=n_clusters, init=init, n_init=len(features), max_iter=10000)
    k_means.fit(features)
    labels = pd.Series(k_means.labels_, name="KMeans")
    return labels


def birch_classification(features, n_clusters=2):
    birch = Birch(n_clusters=n_clusters)
    birch.fit(features)
    labels = pd.Series(birch.labels_, name="Birch")
    return labels


def gaussian_mix_classification(features, n_clusters=2):
    gm = GaussianMixture(n_components=n_clusters)
    labels = gm.fit_predict(features)
    labels = pd.Series(labels, name="Gauss_Mix")
    return labels


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


def normalize_and_center(X):
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    return X


def principal_component_analysis(X, n_compononents='mle'):
    pca = PCA(n_components=n_compononents)
    X = pd.DataFrame(pca.fit_transform(X))
    logging.info("PCA variance ratio")
    logging.info(pca.explained_variance_ratio_)
    return X


def tail_filter(X, col="mean_intensity", demo=False, init_quantile=0.95):
    """ This methods returns the tail of a data distribution according to one of its main component (here "mean_intensity").
    This method is used to remove massive amount of unrelevant data points (here single mollecules) and only keep the extrema of the distribution.
    To do so, an init_quantile filter is applied to remove X% of the dataset. Then the algorithm looks for the first point where the derivative of the kernel distribution goes up to 0.
    This approach allows to select the data points included in the tail of the distribution.

    :param X: pandas.DataFrame
        features to filter
    :param col: str
        name of the column the filter will be based on.
        Default "mean_intensity"
    :param init_quantile:
        Arbitrary quantile to initiate the process. This quantile should cut the distribution at a point where the kernel distribution continously descrease.
        Default 0.95
    :param demo: bool
        if yes, the method will return extra outputs to facilitate plot
        Default False
    :return:
        f_data: pandas DataFrame
        Filtered data
    """
    # Filter at quantile 0.95 to get the highest values (with a band of confidence)
    data_95 = X[X[col] > X[col].quantile(init_quantile)]
    mean_int = data_95[col]

    # calcultate kernel of the distribution. Kernel enable a nice smoothing of the distribution, this avoid local artifactual peaks.
    kde = statsmodels.nonparametric.kde.KDEUnivariate(mean_int)
    kde.fit()
    x_density = kde.support
    density = kde.density
    diff_density = np.diff(density)

    # find zeros in diff
    thresh = None
    for j, d in enumerate(diff_density):
        if j > 0:
            if d <= 0 and diff_density[j + 1] >= 0:
                # Set threshold when diff goes positive for the first time
                thresh = x_density[j + 1]
                break

    # filter data up to the threshold
    f_data = X[X["mean_intensity"] > thresh]

    if demo:
        return f_data, data_95, x_density, diff_density, thresh
    else:
        return f_data


def demo_tail_filter():
    # Get demo files
    dir = "features"
    files = fm.get_files(dir=dir)
    dataset = {}
    for file in files:
        emb = int(file.split('emb')[1].split('.')[0])
        print(file)
        dataset[emb] = fm.get_data_from_file(file)

    # Prepare axis
    fig, ax = plt.subplots(4, 4, figsize=[20, 20])
    i = 0
    for emb, X in dataset.items():

        f_data, data_95, x_density, diff_density, thresh = tail_filter(X, demo=True)
        mean_int = data_95["mean_intensity"]

        # Plot 95% values
        scat = sns.scatterplot(hue="mean_intensity", x='total_intensity', y="area", data=data_95, ax=ax[i][0],
                               legend=False)
        scat.set_title("Embryo {} : {} points (quantile : 95%)".format(emb, len(data_95)))
        # plot distribution and diff
        sns.distplot(mean_int, ax=ax[i][1])
        sns.lineplot(x=x_density[1:], y=diff_density, ax=ax[i][1])

        # Post filtering
        mean_int = f_data["mean_intensity"]
        # Plot final values
        sns.scatterplot(hue="mean_intensity", x='total_intensity', y="area", data=f_data, ax=ax[i][2],
                        legend=False).set_title(
            "Embryo {} : {} points (final)".format(emb, len(f_data)))
        sns.distplot(mean_int, ax=ax[i][3])

        # plot the threshold with a red cross
        if thresh is not None:
            ax[i][1].plot(thresh, 0, marker="x", color='r')
            ax[i][3].plot(thresh, 0, marker="x", color='r')

        i += 1
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    dir = "features"
    files = fm.get_files(dir=dir)
    dataset = {}
    for file in files:
        emb = int(file.split('emb')[1].split('.')[0])
        print(file)
        dataset[emb] = fm.get_data_from_file(file)
        break
    tiff = fm.get_tiff_file(emb)
    for ds in dataset.values():
        f_label= classify(ds)
        viewer = MultiLayerViewer(tiff)
        viewer.plot_imgs(features=f_label)
    plt.show()



