import logging
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, OPTICS, FeatureAgglomeration, \
    KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, bicluster,estimate_bandwidth
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import GaussianMixture

import scripts.file_manager as fm
from scripts.region_3d import extract_molecules
from scripts.performance_monitoring import PerfLogger, Timer

""" This script aims to test multiple clustering models to label data
Warning : some of this models requieres o(n²) memory i.e. for 20k data points it could requiere 400Go RAM. 
This makes somes models like affinity propagation or spectral clustering unrelevent for transcription sites clustering.

"""


# Number of type of mollecules to cluster. Here 'single mollecule' and 'transcription site'
NB_CLUSTER = 2
# Minimum number of 'transcription site'
MIN_SAMPLE = 20

# Minimal ratio of # Transcription size / # single mollecules
# For existing data : emb1 = 0.003179059, emb7 = 0.003564907,emb8 = 0.004652077, emb10 = 0.003987172
MIN_CLUSTER_SIZE = 0.0025


CLUSTERERS = (#(AffinityPropagation(), "Affinity propagation"),
              (AgglomerativeClustering(n_clusters=NB_CLUSTER), "Agglomerative"),
              (Birch(n_clusters=NB_CLUSTER), "Birch"),
              (DBSCAN(min_samples=MIN_SAMPLE), "DBSCAN"),
              (OPTICS(min_samples=MIN_SAMPLE), "OPTICS"),
              (FeatureAgglomeration(n_clusters=NB_CLUSTER), "Feature agglomeration"),
              (KMeans(n_clusters=NB_CLUSTER, n_init=1000, max_iter=10000), "KMeans"),
              (MiniBatchKMeans(n_clusters=NB_CLUSTER, n_init=1000, max_iter=10000 ), "Mini KMeans"),
              (MeanShift(), "MeanShift"),
              #(SpectralClustering(n_clusters=NB_CLUSTER), "Spectral clustering"),
              (GaussianMixture(n_components=NB_CLUSTER), "Gaussian Mixture"))

# TODO : Test others models and parameters
#Test DBSCAN.algorithm = ‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’
#Test AgglomerativeClustering.linkage : {“ward”, “complete”, “average”, “single”}
# test OPTICS
#  .metric : Valid values for metric are:
#     from scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
#     from scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
#  .cluster_method :  “xi” and “dbscan”.
#  .algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
# Test FeatureAgglomeration
# .afinity : [“euclidean”, “l1”, “l2”, “manhattan”, “cosine”, ‘precomputed’]
# .linkage : {“ward”, “complete”, “average”, “single”}
# Test SpectralClustering
# .eigen_solver : {None, ‘arpack’, ‘lobpcg’, or ‘amg’}
# .assign_labels : {‘kmeans’, ‘discretize’},


TRAIN_EMBRYOS = {1: (77, 24221), 7: (82, 23002), 8: (71, 15262), 10: (92, 23074)}


class Multiclusterer:
    def __init__(self, features):
        self.features = features
        self.X, bandwidth, connectivity = self.preprocess_data(features)
        self.clts = ((AgglomerativeClustering(n_clusters=NB_CLUSTER, connectivity=connectivity), "Agglomerative"),
                      (Birch(n_clusters=NB_CLUSTER), "Birch"),
                      (DBSCAN(min_samples=MIN_SAMPLE), "DBSCAN"),
                      (OPTICS(min_samples=MIN_SAMPLE, min_cluster_size=MIN_CLUSTER_SIZE), "OPTICS"),
                      (KMeans(n_clusters=NB_CLUSTER), "KMeans"),
                      (MiniBatchKMeans(n_clusters=NB_CLUSTER), "Mini KMeans"),
                      (MeanShift(bandwidth=bandwidth), "MeanShift"),
                     (SpectralClustering(n_clusters=NB_CLUSTER), "Spectral clustering"))
        self.multiclustering()

    def get_clusters(self):
        return self.clts

    def get_features(self):
        return self.features

    def get_processed_features(self):
        return self.X

    def preprocess_data(self,features):
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(features)

        # estimate bandwidth for mean shift
        bandwidth = estimate_bandwidth(X, quantile=0.5)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        X = pd.DataFrame(X, columns=features.columns)
        return X, bandwidth, connectivity

    def multiclustering(self):
        init = True
        for cluster, name in self.clts:
            y_pred = self.classify(cluster)
            pred = pd.Series(y_pred, name=name)
            if init:
                self.labels = pred
                init = False
            else:
                self.labels = pd.concat([self.labels, pred], axis=1)

    def classify(self, cluster):
        # CLASSIFY DATA
        cluster.fit(self.X)
        if hasattr(cluster, 'labels_'):
            y_pred = cluster.labels_.astype(np.int)
        else:
            y_pred = cluster.predict(self.X)
        return y_pred

    def count_labels(self):
        labels_counts = pd.DataFrame(index=[0,1], columns=self.labels.columns)
        for name in self.labels.columns:
            index = self.labels[name].unique

            d = self.labels[name].value_counts().to_dict()
            labels_counts[name] = labels_counts.index.map(d)
        logging.info(labels_counts)
        return labels_counts

    def export_results(self, file_name = "multiclustering_results"):
        data = pd.concat([self.features,self.labels], axis = 1)
        fm.save_results(data, file_name = file_name, timestamped = True)

    def plot_results(self):
        if not hasattr(self, "labels"):
            self.multiclustering()
        plt.figure(figsize=(9 * 2 + 3, 12.5))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)
        plot_num = 1

        for name in self.labels.columns:
            plt.subplot(2, math.ceil(len(self.clts) / 2), plot_num)
            plt.title(name, size=18)

            try:
                plt.scatter(self.X['area'], self.X['total_intensity'], c=self.labels[name], s=10)
            except Exception as e:
                print(name)
                print(repr(e))
            plt.xticks(())
            plt.yticks(())

            plot_num += 1


#TODO broken, rendre fonctionnel
def demo_multiclustering(X):

    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                      hspace=.01)
    plot_num = 1
    timer = Timer()

    #X, bandwidth, connectivity = preprocess_data(X)

    for cluster, name in CLUSTERERS:
        print('start clustering with {}'.format(name))
        t0 = time.time()
        # CLASSIFY DATA
        cluster.fit(X)
        timer.step()
        print(timer.last_step_duration())
        t1 = time.time()
        if hasattr(cluster, 'labels_'):
            y_pred = cluster.labels_.astype(np.int)
        else:
            y_pred = cluster.predict(X)

        # PLOT RESULT
        plt.subplot(2, math.ceil(len(CLUSTERERS)/2), plot_num)
        plt.title(name, size=18)

        try:
            plt.scatter(X['area'],X['total_intensity'],c=y_pred, s = 10)
        except Exception as e:
            print(name)
            print(repr(e))

        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')

        plot_num += 1


def run_multiclustering(features):
    results = None
    for clst, name in CLUSTERERS:
        print("_" * 80)
        print("Start clustering with {}".format(name))
        try:
            clst.fit(features)
            if hasattr(clst, 'labels_'):
                labels = clst.labels_.astype(np.int)
            else:
                labels = clst.predict(features)

            result = pd.Series(labels, name=name)
            if results is None:
                results = result
            else:
                results = pd.concat([results, result], axis=1)
            print("Success")
        except Exception as e:
            logging.warning("Fail of clustering with {}".format(name))
            logging.warning(repr(e))
            print("Fail")
    return results


def benchmark_clusterers(features, embryo):
    results = run_multiclustering(features)

    labels_counts = pd.DataFrame(index=[0, 1], columns=results.columns)
    try:
        labels_counts["Expected"] = pd.Series({0: TRAIN_EMBRYOS[embryo][1], 1: TRAIN_EMBRYOS[embryo][0]})
    except:
        logging.info("No existing result to compare with.")

    for col in results.columns:
        d = results[col].value_counts().to_dict()
        labels_counts[col] = labels_counts.index.map(d)
    print(labels_counts)
    fm.save_results(labels_counts, file_name="labels_results_embryo_{}.csv".format(embryo), timestamped=True)


def run_all_embryos():
    for embryo in TRAIN_EMBRYOS.keys():
        tiff = fm.get_tiff_file(embryo)
        rf = extract_molecules(tiff)
        features = rf.extract_features()

        try:
            mc = Multiclusterer(features)
            labels_count = mc.count_labels()
            fm.save_results(labels_count, file_name="Labels_counts_multiclustering_emb{}".format(embryo), timestamped=True)
            mc.plot_results()
        except:
            try:
                features = features.drop('centroid_3D', axis=1)
                mc = Multiclusterer(features)
                labels_count = mc.count_labels()
                fm.save_results(labels_count, file_name="Labels_counts_multiclustering_emb{}".format(embryo),
                                timestamped=True)
            except Exception as e:
                print(repr(e))
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    f1 = fm.get_test_features("Result_classif_filter100_2019-08-19-1954")
    #f2 = fm.get_test_features("blob_pipeline_2019-08-19-1348")

    demo_multiclustering(f1)
    #demo_multiclustering(f2)
    plt.show()


