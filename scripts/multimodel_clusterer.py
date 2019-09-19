import logging
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, \
    KMeans, MiniBatchKMeans, MeanShift, SpectralClustering, bicluster, estimate_bandwidth, OPTICS
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import GaussianMixture

import scripts.file_manager as fm
from scripts.classifiers import tail_filter, normalize_and_center, principal_component_analysis
from scripts.file_processing import extract_region_with_cells
from scripts.performance_monitoring import PerfLogger, Timer

""" This script aims to test multiple clustering models to label data
Warning : some of this models requieres o(n²) memory i.e. for 20k data points it could requiere 400Go RAM. 
This makes somes models like affinity propagation or spectral clustering unrelevent for transcription sites clustering.

"""

# Number of type of mollecules to cluster. Here 'single mollecule' and 'transcription site'
NB_CLUSTER = 2
# Minimum number of 'transcription site'
MIN_SAMPLE = 50

# Minimal ratio of # Transcription size / # single mollecules
# For existing data : emb1 = 0.003179059, emb7 = 0.003564907,emb8 = 0.004652077, emb10 = 0.003987172
MIN_CLUSTER_SIZE = 0.0025

CLUSTERERS = (  # (AffinityPropagation(), "Affinity propagation"),
    (AgglomerativeClustering(n_clusters=NB_CLUSTER), "Agglomerative"),
    (Birch(n_clusters=NB_CLUSTER), "Birch"),
    (DBSCAN(min_samples=MIN_SAMPLE), "DBSCAN"),
    #(OPTICS(min_samples=MIN_SAMPLE), "OPTICS"),
    (FeatureAgglomeration(n_clusters=NB_CLUSTER), "Feature agglomeration"),
    (KMeans(n_clusters=NB_CLUSTER, n_init=1000, max_iter=10000), "KMeans"),
    (MiniBatchKMeans(n_clusters=NB_CLUSTER, n_init=1000, max_iter=10000), "Mini KMeans"),
    (MeanShift(), "MeanShift"),
    # (SpectralClustering(n_clusters=NB_CLUSTER), "Spectral clustering"),
    (GaussianMixture(n_components=NB_CLUSTER), "Gaussian Mixture"))

PARAMS = {'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'linkage': ['ward', 'complete', 'average', 'single'],
          'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'braycurtis', 'canberra', 'chebyshev',
                   'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski',
                   'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'],
          'cluster_method': ['xi', 'dbscan'], 'afinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed'],
          'linkage': ['ward', 'complete', 'average', 'single'], 'eigen_solver': [None, 'arpack', 'lobpcg', 'amg'],
          'assign_labels': ['kmeans', 'discretize']}


          # TODO : Test others models and parameters
          # Test DBSCAN.algorithm = ‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’
          # Test AgglomerativeClustering.linkage : {“ward”, “complete”, “average”, “single”}
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

    def preprocess_data(self, features):
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
        labels_counts = pd.DataFrame(index=[0, 1], columns=self.labels.columns)
        for name in self.labels.columns:
            index = self.labels[name].unique

            d = self.labels[name].value_counts().to_dict()
            labels_counts[name] = labels_counts.index.map(d)
        logging.info(labels_counts)
        return labels_counts

    def export_results(self, file_name="multiclustering_results"):
        data = pd.concat([self.features, self.labels], axis=1)
        fm.save_results(data, file_name=file_name, timestamped=True)

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


def demo_multiclustering(X, plt_X = None):
    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    plot_num = 1
    timer = Timer()

    # X, bandwidth, connectivity = preprocess_data(X)

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

        logging.info("prediction results")
        logging.info(pd.Series(y_pred).value_counts())

        # PLOT RESULT
        plt.subplot(2, math.ceil(len(CLUSTERERS) / 2), plot_num)
        plt.title(name, size=18)

        try:
            plt.scatter(plt_X['area'], plt_X['total_intensity'], c=y_pred, s=10)
        except:
            logging.debug("No ploting values")
            try:
                plt.scatter(X['area'], X['total_intensity'], c=y_pred, s=10)
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
        rf = extract_region_with_cells(tiff)
        features = rf.extract_features()

        try:
            mc = Multiclusterer(features)
            labels_count = mc.count_labels()
            fm.save_results(labels_count, file_name="Labels_counts_multiclustering_emb{}".format(embryo),
                            timestamped=True)
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


class Param_tree():
    """ This class uses tree structure to generate permutations of entries. These entries can be parameters for cluster models benchmarking
    exemple :
    if a model has  several parameter value possible :
    distance = [euclidian_distance, other distance]
    n_cluster = [0,1,2]
    >>> tree = Param_tree([[euclidian_distance, other distance],[0,1,2]])
    >>> tree.get_all_permutations()
    [[euclidian_distance, 0],[euclidian_distance, 1],...,[other_distance, 2]]

    Attributes
    ------
    children: <N,M>array
        Collection of values to combine.
    param: array
        Collection of value combined in the node.
    """

    def __init__(self, children=None, param=None):

        self.param = []
        if param is not None:
            # True except for initialisation
            self.param = param

        self.branches = []
        if children is not None:
            for child in children[0]:
                # Add the new param to the list of others.
                child_param = [child]
                child_param.extend(self.param)

                if len(children) == 1:
                    self.branches.append(Param_tree(param=child_param))
                else:
                    new_children = children[1:]
                    self.branches.append(Param_tree(children=new_children, param=child_param))

    def get_all_permutations(self):
        """ Explore the nodes of the tree and return a collection of permutation."""
        params = []
        for branch in self.branches:
            if branch.branches == []:
                params.append(branch.param)
            else:
                params.extend(branch.get_all_permutations())
        return params

def get_isolated_permutations(params):
    base = [par[0] for par in params]
    permutations = {}
    for i, type in enumerate(params):
        for val in type :
            perm = base.copy()
            perm[i]= val
            permutations[val]= perm
    return permutations


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #f1 = fm.get_data_from_file("features_extraction_emb1.csv")
    #f2 = fm.get_test_features("blob_pipeline_2019-08-19-1348")

    dir = "features"
    files = fm.get_files(dir=dir)
    dataset = {}
    for file in files:
        emb = int(file.split('emb')[1].split('.')[0])
        print(file)
        dataset[emb] = fm.get_data_from_file(file)

    for emb, ds in dataset.items():
        print("_"*80)
        print("Embryo {}: {} TS expected".format(emb,TRAIN_EMBRYOS[emb]))
        ds = tail_filter(ds)
        X = normalize_and_center(ds)
        X = principal_component_analysis(X)
        demo_multiclustering(X, plt_X = ds)
    #demo_multiclustering(f2)
    plt.show()

    """ # Test multiparam testing
    from scripts.multimodel_clusterer import get_isolated_permutations
    from sklearn.mixture import GaussianMixture
    import seaborn as sns
    n_components = [2, 3, 5]
    covariance_type = ['tied','full', 'diag', 'spherical']
    tol = [0.00001, 0.001, 0.1]
    reg_covar = [1e-6, 1e-8, 1e-3]
    max_iter = [100, 1000, 10]
    n_init = [1, 10, 100]
    init_params = ['kmeans', 'random']
    random_state = [None, 0]
    warm_start = [True, False]

    dataset = fm.get_data_from_file("features_extraction_emb1.csv")
    #X = dataset[['area', 'total_intensity']]
    X = dataset
    
    params = [n_components, covariance_type, tol, reg_covar, max_iter, n_init, init_params, random_state, warm_start]

    amount = 1
    for lst in params:
        amount += len(lst)

    print(" {} tests to run".format(amount))

    params = get_isolated_permutations(params)

    fig, axs = plt.subplots(amount, 1, figsize=[20, 5 * amount])
    i = 0
    res = X.copy()
    for name, [n_components, covariance_type, tol, reg_covar, max_iter, n_init, init_params, random_state, warm_start] in params.items():
        try:
            #name = "n_components {} / covariance_type {} / tol {} / reg_covar {} / max_iter {} / n_init {} / init_params {} / random_state {} / warm_start {}".format(
            #    n_components, covariance_type, tol, reg_covar, max_iter, n_init, init_params, random_state, warm_start)
            print(name)
            bgm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, tol=tol,
                                  reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params,
                                  weights_init=None, means_init=None, precisions_init=None, random_state=random_state,
                                  warm_start=warm_start)
            labels = bgm.fit_predict(X)
            labels = pd.Series(labels, name=name)
            res = pd.concat([res, labels], axis=1)
            #sns.scatterplot(hue=labels, x='total_intensity', y="area", size="area", sizes=(50, 300), data=res,
            #                ax=axs[i]).set_title(name)
            if i % amount / 5 == 1:
                print(int(i / amount * 100))
        except Exception as e:
            print("issue with {}".format(name))
        finally:
            i += 1
    print(res.head())
    fm.save_results(res, "val_gaussian_mix.csv")"""
