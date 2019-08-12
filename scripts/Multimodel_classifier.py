import logging

import scripts.File_manager as fm
from scripts.Region3D import extract_molecules
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, OPTICS, FeatureAgglomeration, \
    KMeans, MiniBatchKMeans, MeanShift, SpectralClustering
import pandas as pd

NB_CLUSTER = 2
CLUSTERERS = ((AffinityPropagation(), "Affinity propagation"),
              (AgglomerativeClustering(n_clusters=NB_CLUSTER), "Agglomerative"),
              (Birch(n_clusters=NB_CLUSTER), "Birch"),
              (DBSCAN(), "DBSCAN"),
              (OPTICS(), "OPTICS"),
              (FeatureAgglomeration(n_clusters=NB_CLUSTER), "Feature agglomeration"),
              (KMeans(n_clusters=NB_CLUSTER), "KMeans"),
              (MiniBatchKMeans(n_clusters=NB_CLUSTER), "Mini KMeans"),
              (MeanShift(), "MeanShift"),
              (SpectralClustering(n_clusters=NB_CLUSTER), "Spectral clustering"))
# TODO :
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
"""
cluster.AffinityPropagation([damping, …]) 	Perform Affinity Propagation Clustering of data.
cluster.AgglomerativeClustering([…]) 	Agglomerative Clustering
cluster.Birch([threshold, branching_factor, …]) 	Implements the Birch clustering algorithm.
cluster.DBSCAN([eps, min_samples, metric, …]) 	Perform DBSCAN clustering from vector array or distance matrix.
cluster.OPTICS([min_samples, max_eps, …]) 	Estimate clustering structure from vector array
cluster.FeatureAgglomeration([n_clusters, …]) 	Agglomerate features.
cluster.KMeans([n_clusters, init, n_init, …]) 	K-Means clustering
cluster.MiniBatchKMeans([n_clusters, init, …]) 	Mini-Batch K-Means clustering
cluster.MeanShift([bandwidth, seeds, …]) 	Mean shift clustering using a flat kernel.
cluster.SpectralClustering
"""

TRAIN_EMBRYOS = {1: (77, 24221), 7: (82, 23002), 8: (71, 15262), 10: (92, 23074)}


def run_all_embryos():
    for embryo in TRAIN_EMBRYOS.keys():
        tiff = fm.get_tiff_file(embryo)
        rf = extract_molecules(tiff)
        features = rf.extract_features()
        benchmark_clusterers(features, embryo)

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
    fm.save_results(labels_counts, file_name="labels_results_embryo_{}.csv".format(embryo), timestamped = True)


def run_multiclustering(features):
    results = None
    for clst, name in CLUSTERERS:
        print("_"*80)
        print("Start clustering with {}".format(name))
        try:
            clst.fit(features)
            labels = pd.Series(clst.labels_, name=name)
            if results is None:
                results = labels
            else:
                results = pd.concat([results, labels],axis=1)
            print("Success")
        except Exception as e:
            logging.warning("Fail of clustering with {}".format(name))
            logging.warning(repr(e))
            print("Fail")
    return results

"""
class MultimodelClustering:
    def __init__(self):
        pass


    def benchmark_clusterers(self):
        for embryo in TRAIN_EMBRYOS.keys():
            tiff = fm.get_tiff_file(embryo)
            rf = extract_molecules(tiff)
            features = rf.extract_features()
            results = self.run_clustering(features)
            labels_counts = pd.DataFrame(index = [0,1], columns=results.columns)
            labels_counts["Expected"] = pd.Series({0:TRAIN_EMBRYOS[embryo][1], 1:TRAIN_EMBRYOS[embryo][0]})
            for col in results.columns:
                d = results[col].value_counts().to_dict
                labels_counts[col] = labels_counts.index.map(d)
            print(labels_counts)
            import datetime
            labels_counts.to_csv("labels_results_embryo_{0}_{1}".format({0:embryo, 1:datetime.now()}))


    def run_clustering(self, features):
        results = None
        for clst, name in CLUSTERERS:
            try:
                clst.fit(features)
                labels = pd.Series(clst.labels_, name = name)
                if results is None:
                    results = labels
                else :
                    results = pd.concat(results, labels)
            except Exception as e:
                logging.warning("Fail of clustering with {]".format(name))
                logging.warning(repr(e))
        return results



    def get_best_classifiers(self, ratio):
        results = []
        for clf, name in CLASSIFIERS:
            try:
                clf.fit(self.x_train, self.y_train)
                pred = clf.predict(self.x_test)
                if name in ["Random Forest Regressor", "MPLR"]:
                    pred = pred.round()
                score = metrics.accuracy_score(self.y_test, pred)
                logging.info(name + " score: " + str(score * 100) + " %")
                results.append((clf, name, score))
            except Exception as e:
                logging.warning("Fail in training " + name)
                logging.warning(repr(e))
                traceback.print_tb(e.__traceback__)
                t.sleep(2)

        logging.info("Models tested : ")
        logging.info(len(results))
        classifiers = [result for result in results if result[2] > ratio]
        logging.info("Models retained : ")
        logging.info(len(classifiers))
        return classifiers

    def split_data(self, df):
        # TODO définir y et l'enlever du dataset
        y = df["Type"]
        x = df.drop("Type", axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)
        return x_train, x_test, y_train, y_test

    def prep_data(self, data):
        # TODO Prepare data before feature extaction
        return df_vect

    def predict_all(self, df):
        preds = pd.DataFrame()
        for clf, name, score in self.classifiers:
            try:
                preds = pd.concat([preds, pd.DataFrame(clf.predict(df), columns=[name])], axis=1)
            except Exception as e:
                logging.warning("=" * 80)
                logging.warning("Fail in predicting with " + name)
                logging.warning("_" * 80)
                logging.warning(repr(e))
        logging.info("Predictions")
        logging.info("=" * 80)
        logging.info(preds)

        # TODO compute classification score
        scores = compute_scores(preds)
        # TODO Retains the best predictions
        df_res = self.method_max(scores)

        return df_res

    def labelise(self, input, transcript=True, print_results=True):
        if isinstance(input, str):
            try:
                raw_df, df_x = get_data(input)
            except Exception as e:
                raise e
        elif isinstance(input, pd.DataFrame):
            logging.info(input.columns)
            df_x = input.reset_index().drop('index', axis=1)
            raw_df = df_x

        else:
            raise ValueError("Input must be String (path) or pd.DataFrame")

        if any(col in df_x.columns for col in ('Libellé', 'LibellÃ©')):
            df_vect = self.prep_data(data=df_x)
        else:
            df_vect = df_x

        if 'Type' in df_vect.columns:
            df_vect = df_vect.drop("Type", axis=1)

        df_pred = self.predict_all(df_vect)

        if print_results:
            self.print_results(df_pred)

        if transcript:
            df_pred = self.transcript_results(df_pred, raw_df)
        return df_pred

    def method_max(self, scores):
        pred = scores.idxmax(axis=1)
        total_preds = scores.iloc[0].sum()
        confusion_scores = []
        for row in scores.index.values:
            vals = scores.iloc[row]
            amount_of_pred = vals.loc[pred[row]]
            confusion_scores.append(amount_of_pred / total_preds)
        confusion_scores = pd.Series(confusion_scores)

        pred = pd.concat([pred, confusion_scores], axis=1)
        pred.columns = ["Predictions", "Confusion score"]

        logging.info("\n Result of Max Method")
        logging.info("=" * 80)
        logging.info(pred)
        return pred

    def print_results(self, df_res):
        try:
            print('Results on the test set:')
            print("Accuracy : " + str(accuracy_score(self.y_test, df_res["Predictions"])))
            print(classification_report(self.y_test, df_res["Predictions"]))
        except Exception as e:
            logging.warning("Impossible to print results of classification")
            logging.warning(repr(e))
"""

if __name__ == '__main__':
    features = fm.get_test_features()
    features = features.drop('centroid_3D', axis=1)
    benchmark_clusterers(features,11)