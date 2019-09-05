import time

import pandas as pd
from math import sqrt
from scipy.signal import argrelextrema, hilbert, find_peaks
from skimage import io, color, measure
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon

from scripts.file_viewer import MultiSliceViewer
import os
import sys
import scripts.file_manager as fm
import logging
import operator
# ________________________________________________
# Imports for label region
import matplotlib.patches as mpatches

from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage.measure import regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import seaborn as sns

# ________________________________________________
# Imports for local maximas
from skimage.measure import label
from skimage import color
from skimage.morphology import extrema
# ________________________________________________

# Import for Fundamental matrix estimation¶
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
# ________________________________________________

# Import for blob detection
from skimage.feature import blob_log

# Import for rescale intensity
from skimage.exposure import rescale_intensity, histogram

# Import for clustering
from sklearn.cluster import KMeans, Birch

from scripts.performance_monitoring import PerfLogger

"""
OBJ : Compter le nombre de TS et single mol par noyau
Règle : Max 2 TS par noyau

"""
# todo : replace prints by loggings

FILE_NAME = "C10DsRedlessxYw_emb11_Center_Out.tif"
DATA_PATH = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data"

PATH_TO_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PATH_TO_ROOT_DIR = os.path.normpath(os.path.join(PATH_TO_CURRENT_DIR, '..'))
sys.path.append(PATH_TO_ROOT_DIR)

DATA_PATH = os.path.join(PATH_TO_ROOT_DIR, 'data')

# format of embryios data : {id : (TS, single mol)}
EMBRYOS = {1: (77, 24221), 7: (82, 23002), 8: (71, 15262), 10: (92, 23074)}
COLORS = {1: "#FF0000", 2: "#FF00FF", 3: "#0000FF"}


def extract_channels(im):
    one = im.copy()
    one[:, :, :, 1:] = 0
    two = im.copy()
    two[:, :, :, 2::-2] = 0
    three = im.copy()
    three[:, :, :, :2] = 0
    return one, two, three


def convert_HSV(im):
    """
    Hue Saturation Value color space conversion

    :param im: RGB image
    :return: HSV image
    """
    if len(im.shape) == 4:
        im_hsv = np.empty(im.shape)
        for i in np.arange(len(im)):
            im_hsv[i] = color.rgb2hsv(im[i])
        print(im_hsv.shape)

    else:
        im_hsv = color.rgb2hsv(im)
    return im_hsv


def convert_grey(im):
    """
    grey-scale color space conversion

    :param im: RGB image
    :return: black & white image
    """
    if len(im.shape) == 4:
        im_grey = np.empty(im.shape)
        for i in np.arange(len(im)):
            im_grey[i] = color.rgb2grey(im[i])
        print(im_grey.shape)

    else:
        im_grey = color.rgb2grey(im)
    return im_grey


def convert_HED(im):
    """
    Haematoxylin-Eosin-DAB (HED) color space conversion

    :param im: image RGB
    :return: image HED
    """
    if len(im.shape) == 4:
        im_hed = np.empty(im.shape)
        for i in np.arange(len(im)):
            im_hed[i] = color.rgb2hed(im[i])
        print(im_hed.shape)

    else:
        im_hed = color.rgb2hed(im)
    return im_hed



def add_img(image, axs, cmap='gnuplot2', col=0, row=None, title="Undefined"):
    type = None
    try:
        shape = axs.shape
        if len(shape) > 1:
            type = "matrix"
        elif len(shape) == 1:
            type = 'row'
        else:
            raise TypeError("Some issues occured during type definition")
    except:
        try:
            shape = len(axs)
            type = 'row'
        except:
            try:
                axs.set_title(title)
                type = 'single'
            except:
                raise TypeError("Fail in ploting image : Axis is not properly setted")

    if type == 'matrix':
        axs[row][col].imshow(image, cmap=cmap)
        axs[row][col].set_title(title)
        if col == shape[1] - 1:
            col = 0
            row += 1
        else:
            col += 1

    elif type == 'row':
        axs[col].imshow(image, cmap=cmap)
        axs[col].set_title(title)
        col += 1
    elif type == 'single':
        axs.imshow(image, cmap=cmap)
        axs.set_title(title)

    return row, col



# todo : Try ridges operator to enhance edges : https://scikit-image.org/docs/dev/auto_examples/edges/plot_ridge_filter.html#sphx-glr-auto-examples-edges-plot-ridge-filter-py

# todo : try hysteris detection : https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html#sphx-glr-auto-examples-filters-plot-hysteresis-py

# todo : Try entropy to enhace edges and facilitate detection https://scikit-image.org/docs/dev/auto_examples/filters/plot_entropy.html#sphx-glr-auto-examples-filters-plot-entropy-py

# todo : try template matching https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html#sphx-glr-auto-examples-features-detection-plot-template-py

# todo : try img_to_graph as (additionnal?) feature extraction prior to classification. Consider using sklearn.pipeline.


def local_maximas(img, h=None):
    if h:
        ncols = 3
    else:
        ncols = 2
    fig, axs = plt.subplots(ncols=ncols, nrows=1, figsize=(12, 4))
    col, row = (0, 0)
    row, col = add_img(img, axs, col=col, row=row, title="Initiale")

    # We find all local maxima
    local_maxima = extrema.local_maxima(img)
    label_maxima = label(local_maxima)
    overlay = color.label2rgb(label_maxima, img, alpha=0.7, bg_label=0,
                              bg_color=None, colors=[(1, 0, 0)])
    row, col = add_img(overlay, axs, col=col, row=row, title="local maximas")
    logging.info("# of regions : " + str(len(regionprops(label_maxima))))
    if h:
        h_maxima = extrema.h_maxima(img, h)
        label_h_maxima = label(h_maxima)
        overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7, bg_label=0,
                                    bg_color=None, colors=[(1, 0, 0)])
        row, col = add_img(overlay_h, axs, col=col, row=row, title="local maximas thresholded")
        logging.info("# of regions after thresholding : " + str(len(regionprops(label_h_maxima))))
    return img, label_maxima, label_h_maxima



# Unused yet
def extract_intensity_features_from_region(region, im):
    minr, minc, maxr, maxc = region.bbox
    intensities = im[minr: maxr, minc: maxc]
    int_max = np.amax(intensities)
    int_min = np.min(intensities)
    overall_int = np.sum(intensities)
    mean_intensity = np.mean(intensities)
    return [int_min, int_max, overall_int, mean_intensity]


# ________________________________________________
#              PERSISTANCE TRACKING
# ________________________________________________

# todo : Do not work with maximas map
def fundamental_matrix_estimation(im1, im2):
    descriptor_extractor = ORB()

    # process img 1
    descriptor_extractor.detect_and_extract(im1)
    keypoints_1 = descriptor_extractor.keypoints
    descriptors_1 = descriptor_extractor.descriptors

    # process img 2
    descriptor_extractor.detect_and_extract(im2)
    keypoints_2 = descriptor_extractor.keypoints
    descriptors_2 = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_1, descriptors_2,
                                cross_check=True)

    # Estimate the epipolar geometry between the left and right image.

    model, inliers = ransac((keypoints_1[matches[:, 0]],
                             keypoints_2[matches[:, 1]]),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=1, max_trials=5000)

    inlier_keypoints_1 = keypoints_1[matches[inliers, 0]]
    inlier_keypoints_2 = keypoints_2[matches[inliers, 1]]

    logging.info("Number of matches:", matches.shape[0])
    logging.info("Number of inliers:", inliers.sum())

    # Compare estimated sparse disparities to the dense ground-truth disparities.

    disp = inlier_keypoints_1[:, 1] - inlier_keypoints_2[:, 1]
    disp_coords = np.round(inlier_keypoints_1).astype(np.int64)
    disp_idxs = np.ravel_multi_index(disp_coords.T, im1.shape)
    disp_error = np.abs(im1.ravel()[disp_idxs] - disp)
    disp_error = disp_error[np.isfinite(disp_error)]

    # Visualize the results.

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], im1, im2, keypoints_1, keypoints_2,
                 matches[inliers], only_matches=True)
    ax[0].axis("off")
    ax[0].set_title("Inlier correspondences")

    ax[1].hist(disp_error)
    ax[1].set_title("Histogram of disparity errors")




if __name__ == '__main__':
    embryos = fm.get_embryos()
    for emb in embryos:
        ch3 = fm.get_tiff_file(emb)[:,:,:,2]
        contours = find_cells_contours(ch3)
        label_img, bin = label_contours(contours,ch3[0].shape)
        fig, ax = plt.subplots()
        ax.imshow(bin)


    """
    ch1 = fm.get_tiff_file(1)[:,:,:,2]
    contours = find_cells_contours(ch1)
    bin = np.zeros(ch1[0].shape)
    for c in contours :
        rr, cc = polygon(c[:, 0], c[:, 1], bin.shape)
        bin[rr, cc] = 1

    plt.imshow(bin)"""
    plt.show()
    """
    rscl_img = rscl_intensity(ch1)
    viewer = MultiSliceViewer(rscl_img)
    all_blobs = np.array([blob_detection(img) for img in rscl_img])
    viewer.plot_imgs(blobs=all_blobs)
    plt.show()"""


    """img = ch1[14]
    label_img = label_filter(img, filter=0.1)[0]
    regions = region_properties(label_img,img, properties=['coords'])

    r=regions.iloc[45]

    coords = []
    for i, region in regions.iterrows():
        for pixel in region['coords'] :
            if len(coords)==0:
                coords = [pixel]
            else :
                coords = np.append(coords, [pixel] , axis = 0)


    coords = np.array([[coord[0], coord[1]] for coord in coords])

    coord = np.array([[coord[0], coord[1]] for coord in r['coords']])

    coord = np.append(coord, [[32, 1050]], axis = 0)
    print(coords[:,0])
    for x in coord :
        test = np.all(coords == x, axis=1)
        test_2= np.argwhere(test)
        test_3 = np.argwhere(np.logical_and(coords[:, 0] == x[0], coords[:, 1] == x[1]))
        print(test)
        print(test_2)
        print(test_3)
        print("-"*15)

    test = [np.all(coords == x, axis=1) for x in coord]
    #TODO:
    # Ajouter un % d'inclusion dans une région.
    # Utiliser enumerate pour identifier la region ( cellule ) dans laquelle se situe la mollecule
    # Labelliser les regions (molecules) en fonction du fait qu'elles soient dans une autre région (cellule)
    # ajouter cette labellisation aux features des régions3D

    print(test)"""
    #df_coords = pd.DataFrame(coord, columns=["coord"])
    #mask = df_coords.isin({'coord': coords})

    #demo_regions(ch3, label_img, min_area=2)
    #plt.show()


    """features = fm.get_test_features()
    features = features.drop(["centroid_3D"], axis = 1)
    kmeans, labels = kmeans_classification(features)
    labels.to_csv("Labels_cells.csv", index=False)
    result = pd.concat([features, labels], axis=1)
    result.to_csv("Features_&_Labels_cells.csv", index=False)"""

    """
    df = pd.read_csv("Features_&_Labels_cells.csv")
    df = df.drop(['min_intensity', '0', 'id ','extent'], axis = 1)
    #df.plot.density()
    for col in df.columns:
        s = df[col]
        s.plot.density()
        plt.show()
    """
