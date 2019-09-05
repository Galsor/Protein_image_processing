import time

import pandas as pd
from math import sqrt
from scipy.signal import argrelextrema, hilbert, find_peaks
from skimage import io, color, measure
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon

from scripts.multi_slice_viewer import MultiSliceViewer
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


def plot_img(image, cmap='gnuplot2', title="Undefined title"):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    fig.suptitle(title)

    if isinstance(image[0, 0], bool):
        # for binarys ploting
        ax.imshow(image, cmap='Greys', interpolation='nearest')
    else:
        ax.imshow(image, cmap=cmap)


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


# ________________________________________________
#               PATTERN DETECTION
# ________________________________________________

def find_cells_contours(imgs, window=500, intensity_band=(400, 900), smooth=50, demo=False):
    all_peaks = detect_intensity_peaks(imgs, window=window, intensity_band=intensity_band, smooth=smooth)

    def compute_area(contour):
        x = contour[:, 0]
        y = contour[:, 1]
        area = 0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
        area = np.abs(area)
        return area

    selected_img_idx = None
    res = 0
    a_c_max = 0
    for i, img in enumerate(imgs):
        if len(all_peaks[i]) == 0:
            continue
        logging.info(" ---- Image {} ----".format(i))
        # find contours
        contours = measure.find_contours(img, all_peaks[i][0], fully_connected='high', positive_orientation='high')
        # select significant contours (length > 200)
        contours = [c for c in contours if len(c) > 200]
        # get length for each contours
        contours_len = [len(c) for c in contours]

        # compute area related to each contours
        areas = []
        for c in contours:
            area = compute_area(c)
            areas.append(area)

        # compute area/contour ratio (total areas / total contours)
        # high value of ratio means that contours are nicely embedding the cell (close from ellipic form)
        a_c_ratio = np.sum(areas) / np.sum(contours_len)

        # Select only images where regions has reasonnable size
        test = np.max(areas)
        if np.max(areas) < 90000 and a_c_ratio > a_c_max:
            a_c_max = a_c_ratio
            res = contours
            selected_img_idx = i
    if demo:
        return res, selected_img_idx
    else:
        return res


def demo_find_cells_contours(tiff):
    imgs = tiff[:, :, :, 2]

    contours, i = find_cells_contours(imgs, demo=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].imshow(imgs[i], cmap='gnuplot2')
    ax[0].set_title("Image {} ".format(i))
    for n, contour in enumerate(contours):
        if len(contour) > 200:
            ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2)
    hist, hist_centers = histogram(imgs[i])
    ax[1].plot(hist_centers, hist, lw=2)
    ax[1].set_title('histogram of gray values')


def detect_intensity_peaks(tiff, window=500, intensity_band=(400, 900), smooth=50, demo=False):
    try:
        dim = len(tiff.shape)
    except:
        raise TypeError("Wrong type of image input for intensity peaks detection")

    if dim == 4:
        imgs = tiff[:, :, :, 2]
    elif dim == 3:
        imgs = tiff
    elif dim == 2:
        imgs = [tiff]
    else:
        raise ValueError("Image dimension is not bounded between 2 ('single image') and 4  ('all channel tiff file')")

    if not isinstance(intensity_band, tuple) and not isinstance(intensity_band, int):
        raise TypeError("Wrong type of input input for intensity band. Please use number  or tuple<int>")

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    hists = []
    all_peaks = []
    for i, img in enumerate(imgs):
        hist, hist_centers = histogram(img)
        # smooth local variations
        avg_hist = moving_average(hist, smooth)
        peaks, _ = find_peaks(avg_hist, distance=window, height=intensity_band)
        logging.info("img {} : {}".format(i, peaks))
        hists.append(hist)
        all_peaks.append(peaks)

    # convert the list in numpy matrix
    all_peaks = np.array(all_peaks)

    if demo:
        return imgs, hists, all_peaks
    else:
        return all_peaks


def demo_detect_intensity_peaks(tiff):
    fig, axs = plt.subplots(4, 10)
    imgs, hist, peaks = detect_intensity_peaks(tiff, demo=True)
    for i, img in enumerate(imgs):
        p_row = int(i / 10)
        p_col = int(i % 10)
        ax = axs[p_row][p_col]
        ax.set_axis_off()
        ax.plot(hist[i], lw=2)
        ax.plot(peaks[i], hist[i][peaks[i]], "x")
        ax.set_title(i)


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


# ________________________________________________
# BLOBS DETECTION

def blob_extraction(tiff):
    rscl_img = rscl_intensity(tiff)
    logging.info("End rescaling")

    blobs = [blob_detection(im, log_scale=True) for im in rscl_img]
    logging.info("End blob detection")

    # compute radii
    for blobs_layer in blobs:
        if len(blobs_layer) > 1:
            blobs_layer[:, 2] = blobs_layer[:, 2] * sqrt(2)
    logging.info("End compute blob radii")
    fm.save_as_pickle([blobs, rscl_img], file_name="data_test_blob_detect")
    return blobs, rscl_img


def blob_detection(img, log_scale=True):
    blobs = blob_log(img, log_scale=log_scale)
    return blobs


def demo_blobs(img):
    rscl_intensity(img)
    blobs = blob_detection(img)
    # Compute radii in the 3rd column.
    blobs[:, 2] = blobs[:, 2] * sqrt(2)

    fig, ax = plt.subplots(1, 1, figsize=(9, 3))

    ax.set_title("{} Blobs detected".format(len(blobs)))
    ax.imshow(img)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='blue', linewidth=2, fill=False)
        ax.add_patch(c)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


# ________________________________________________
# REGIONS PROPERTIES

COL_DTYPES = {
    'area': int,
    'bbox': int,
    'bbox_area': int,
    'moments_central': float,
    'centroid': int,
    'convex_area': int,
    'convex_image': object,
    'coords': object,
    'eccentricity': float,
    'equivalent_diameter': float,
    'euler_number': int,
    'extent': float,
    'filled_area': int,
    'filled_image': object,
    'moments_hu': float,
    'image': object,
    'inertia_tensor': float,
    'inertia_tensor_eigvals': float,
    'intensity_image': object,
    'label': int,
    'local_centroid': int,
    'major_axis_length': float,
    'max_intensity': float,
    'mean_intensity': float,
    'min_intensity': float,
    'minor_axis_length': float,
    'moments': float,
    'moments_normalized': float,
    'orientation': float,
    'perimeter': float,
    'slice': object,
    'solidity': float,
    'weighted_moments_central': float,
    'weighted_centroid': int,
    'weighted_moments_hu': float,
    'weighted_local_centroid': int,
    'weighted_moments': int,
    'weighted_moments_normalized': float
}

PROPS = {
    'Area': 'area',
    'BoundingBox': 'bbox',
    'BoundingBoxArea': 'bbox_area',
    'CentralMoments': 'moments_central',
    'Centroid': 'centroid',
    'ConvexArea': 'convex_area',
    # 'ConvexHull',
    'ConvexImage': 'convex_image',
    'Coordinates': 'coords',
    'Eccentricity': 'eccentricity',
    'EquivDiameter': 'equivalent_diameter',
    'EulerNumber': 'euler_number',
    'Extent': 'extent',
    # 'Extrema',
    'FilledArea': 'filled_area',
    'FilledImage': 'filled_image',
    'HuMoments': 'moments_hu',
    'Image': 'image',
    'InertiaTensor': 'inertia_tensor',
    'InertiaTensorEigvals': 'inertia_tensor_eigvals',
    'IntensityImage': 'intensity_image',
    'Label': 'label',
    'LocalCentroid': 'local_centroid',
    'MajorAxisLength': 'major_axis_length',
    'MaxIntensity': 'max_intensity',
    'MeanIntensity': 'mean_intensity',
    'MinIntensity': 'min_intensity',
    'MinorAxisLength': 'minor_axis_length',
    'Moments': 'moments',
    'NormalizedMoments': 'moments_normalized',
    'Orientation': 'orientation',
    'Perimeter': 'perimeter',
    # 'PixelIdxList',
    # 'PixelList',
    'Slice': 'slice',
    'Solidity': 'solidity',
    # 'SubarrayIdx'
    'WeightedCentralMoments': 'weighted_moments_central',
    'WeightedCentroid': 'weighted_centroid',
    'WeightedHuMoments': 'weighted_moments_hu',
    'WeightedLocalCentroid': 'weighted_local_centroid',
    'WeightedMoments': 'weighted_moments',
    'WeightedNormalizedMoments': 'weighted_moments_normalized'
}

OBJECT_COLUMNS = {
    'image', 'coords', 'convex_image', 'slice',
    'filled_image', 'intensity_image'
}


def region_properties(label_image, image=None, min_area=1, properties=None, separator='-'):
    """
    Convert image region properties  and list them into a column dictionary.

    Parameters
    ----------
    label_image : (N, M) ndarray
        Labeled input image. Labels with value 0 are ignored.
        .. versionchanged:: 0.14.1
            Previously, ``label_image`` was processed by ``numpy.squeeze`` and
            so any number of singleton dimensions was allowed. This resulted in
            inconsistent handling of images with singleton dimensions. To
            recover the old behaviour, use
            ``regionprops(np.squeeze(label_image), ...)``.
    min_area : int, optional
        Minimum area size of regions
        Default is 1.
    image : (N, M) ndarray, optional
        Intensity (i.e., input) image with same size as labeled image.
        Default is None.
    properties : tuple or list of str, optional
        Properties that will be included in the resulting dictionary
        For a list of available properties, please see :func:`regionprops`.
        Users should remember to add "label" to keep track of region
        identities.
    separator : str, optional
        For non-scalar properties not listed in OBJECT_COLUMNS, each element
        will appear in its own column, with the index of that element separated
        from the property name by this separator. For example, the inertia
        tensor of a 2D region will appear in four columns:
        ``inertia_tensor-0-0``, ``inertia_tensor-0-1``, ``inertia_tensor-1-0``,
        and ``inertia_tensor-1-1`` (where the separator is ``-``).
        Object columns are those that cannot be split in this way because the
        number of columns would change depending on the object. For example,
        ``image`` and ``coords``.
    Returns
    -------
    regions : (N,) list
        List of RegionProperties objects as returned by :func:`regionprops`.
    out_dict : dict
        Dictionary mapping property names to an array of values of that
        property, one value per region. This dictionary can be used as input to
        pandas ``DataFrame`` to map property names to columns in the frame and
        regions to rows.
    Notes
    -----
    Each column contains either a scalar property, an object property, or an
    element in a multidimensional array.
    Properties with scalar values for each region, such as "eccentricity", will
    appear as a float or int array with that property name as key.
    Multidimensional properties *of fixed size* for a given image dimension,
    such as "centroid" (every centroid will have three elements in a 3D image,
    no matter the region size), will be split into that many columns, with the
    name {property_name}{separator}{element_num} (for 1D properties),
    {property_name}{separator}{elem_num0}{separator}{elem_num1} (for 2D
    properties), and so on.
    For multidimensional properties that don't have a fixed size, such as
    "image" (the image of a region varies in size depending on the region
    size), an object array will be used, with the corresponding property name
    as the key."""
    if image is None:
        regions = regionprops(label_image)
    else:
        try:
            regions = regionprops(label_image, image)
        except Exception as err:
            raise err

    # Select only the regions up to the min area criteria
    regions = [region for region in regions if region.area > min_area]

    if not properties:
        properties = PROPS.values()

    r_properties = {}
    n = len(regions)
    for prop in properties:
        try:
            dtype = COL_DTYPES[prop]
            column_buffer = np.zeros(n, dtype=dtype)
            r = regions[0][prop]

            # scalars and objects are dedicated one column per prop
            # array properties are raveled into multiple columns
            # for more info, refer to notes 1
            if np.isscalar(r) or prop in OBJECT_COLUMNS:
                for i in range(n):
                    column_buffer[i] = regions[i][prop]
                r_properties[prop] = np.copy(column_buffer)
            else:
                if isinstance(r, np.ndarray):
                    shape = r.shape
                else:
                    shape = (len(r),)

                for ind in np.ndindex(shape):
                    for k in range(n):
                        loc = ind if len(ind) > 1 else ind[0]
                        column_buffer[k] = regions[k][prop][loc]
                    modified_prop = separator.join(map(str, (prop,) + ind))
                    r_properties[modified_prop] = np.copy(column_buffer)
        except Exception as err:
            logging.debug("Error with : " + prop)
            logging.debug(repr(err))
    r_properties = pd.DataFrame(r_properties)
    return r_properties


def demo_regions(image, label_image, show_image=None, min_area=4, title="Demo of region detection"):
    # Compute regions properties
    if show_image is None:
        show_image = image

    props = region_properties(label_image, image=image, min_area=min_area,
                              properties=['extent', 'max_intensity', 'area', "mean_intensity", "bbox"])

    fig, axs = plt.subplots(ncols=2, figsize=(10, 6))

    def draw_rectangles(properties, picked_region=None):
        axs[0].clear()
        axs[0].imshow(show_image, cmap='gnuplot2')
        logging.debug(picked_region)
        if picked_region is not None:
            picked_minr = picked_region['bbox-0'].values
            picked_minc = picked_region['bbox-1'].values
            picked_maxr = picked_region['bbox-2'].values
            picked_maxc = picked_region['bbox-3'].values
            picked_bbox = [picked_minr, picked_minc, picked_maxr, picked_maxc]
            logging.debug(picked_bbox)
        for index, row in properties.iterrows():  # draw rectangle around segmented coins
            minr = properties['bbox-0'].iloc[index]
            minc = properties['bbox-1'].iloc[index]
            maxr = properties['bbox-2'].iloc[index]
            maxc = properties['bbox-3'].iloc[index]
            bbox = [minr, minc, maxr, maxc]

            if picked_region is not None and picked_bbox == bbox:
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2, picker=True)
            else:
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='blue', linewidth=2, picker=True)
            axs[0].add_patch(rect)

    draw_rectangles(props)

    logging.debug(props.shape)
    logging.debug(props.head())

    points = axs[1].scatter(x=props['max_intensity'], y=props["mean_intensity"], facecolors=["C0"] * len(props),
                            edgecolors=["C0"] * len(props), picker=True)
    fc = points.get_facecolors()

    def change_point_color(indexes):
        for i in indexes:  # might be more than one point if ambiguous click
            new_fc = fc.copy()
            new_fc[i, :] = (1, 0, 0, 1)
            points.set_facecolors(new_fc)
            points.set_edgecolors(new_fc)
        fig.canvas.draw_idle()

    axs[1].set_title("{} regions detected".format(props.shape[0]))
    axs[0].set_axis_off()
    fig.suptitle(title, fontsize=14, fontweight='bold')

    def onpick(event):
        logging.debug("Fire")
        try:
            ind = event.ind
            if len(ind) > 1:
                ind = [ind[0]]
        except:
            if isinstance(event.artist, mpatches.Rectangle):
                minc, minr = event.artist.get_xy()
                ind = props.loc[props['bbox-0'] == minr].loc[props['bbox-1'] == minc].index
                if len(ind) > 1:
                    logging.warning("warning {} values in ind while clicking on rectangles".format(len(ind)))

        change_point_color(ind)
        region_props_picked = props.iloc[ind]
        draw_rectangles(props, region_props_picked)

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.tight_layout()

    return props


# ________________________________________________
#               IMAGE PREPROCESSING
# ________________________________________________

def rscl_intensity(img, low_perc=1, high_perc=99):
    p_start, p_end = np.percentile(img, (low_perc, high_perc))
    if len(img.shape) == 3:
        rscl_img = np.array([rescale_intensity(im, in_range=(p_start, p_end)) for im in img])
    elif len(img.shape) == 2:
        rscl_img = rescale_intensity(img, in_range=(p_start, p_end))
    else:
        raise Exception("Some problem occures while processing rscl_intensity. Please check the dimension of img used")
    return rscl_img


def define_filter_value(image, filter, window_size=5, k=0.2):
    if not filter:
        thresh = 100
    elif isinstance(filter, int):
        thresh = filter
    elif filter < 1 and filter > 0:
        max = np.amax(image)
        min = np.amin(image)
        intensity_band = max - min
        thresh = intensity_band * filter
    elif filter == "otsu":
        thresh = threshold_otsu(image)
    elif filter == "niblack":
        thresh = threshold_niblack(image, window_size=window_size, k=k)
    elif filter == "sauvola":
        thresh = threshold_sauvola(image, window_size=window_size)

    return thresh


def label_filter(image, filter=None, window_size=5, k=0.2, close_square=2):
    """ Apply intensity filter

    :param image: (N, M) ndarray
        Input image.
    :param filter: str, int, optional
        Type of filter use. Possibles :
            -  Any integer value
            - "otsu",
            - "niblack",
            - "sauvola"
        Default : 100
    :param window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
        Default : 5
    :param k: float, optional
        Value of parameter k in niblack threshold formula.
        Default : 0.2
    :return:
        label_image : array, same shape and type as image
            The result of the morphological closing
        image_label_overlay : : array, same shape and type as image
            The overlay of the original image with the label_image
    """

    # compute filter value
    if isinstance(filter, tuple):
        thresh_min = define_filter_value(image, filter[0], window_size, k)
        thresh_max = define_filter_value(image, filter[1], window_size, k)
    else:
        thresh_min = define_filter_value(image, filter, window_size, k)
        thresh_max = None

    logging.info("Threshold : ")
    logging.info(thresh_min)

    # apply filter
    if thresh_max is None:
        binary = image > thresh_min
    else:
        binary = np.logical_and(image > thresh_min, image < thresh_max)

    # close blanks
    bw = closing(binary, square(close_square))

    # label image regions
    label_image = label(bw)

    return label_image, binary


def demo_label_filter(image):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    axs[0][0].imshow(image)
    axs[0][0].set_title("Original")
    filters = [None, 250, "otsu", "niblack", "sauvola"]
    for i in range(5):
        label_image, image_label_overlay = label_filter(image, filter=filters[i])
        row = int((i + 1) / 3)
        col = (i + 1) % 3
        axs[row][col].imshow(image_label_overlay, cmap="gray")
        axs[row][col].set_axis_off()
        axs[row][col].set_title(filters[i])

    plt.tight_layout()
    plt.show()


def label_blob(img, blobs, filter=None, window_size=5, k=0.2):
    # Apply threshold
    thresh = define_filter_value(img, filter, window_size, k)
    bin = img > thresh

    blob_bin = np.zeros(img.shape)

    try:
        for blob in blobs:
            # find coordinates of the region with respect of the image edges
            min_row = int(blob[0] - blob[2]) if (int(blob[0] - blob[2]) > 0) else 0
            max_row = int(blob[0] + blob[2]) if (int(blob[0] + blob[2]) < len(img)) else len(img) - 1
            min_col = int(blob[1] - blob[2]) if (int(blob[1] - blob[2]) > 0) else 0
            max_col = int(blob[1] + blob[2]) if (int(blob[1] + blob[2]) < len(img)) else len(img) - 1

            # Select the label_image slice corresponding to the blob and add it
            # This phase remove labeled regions
            blob_bin[min_row:max_row, min_col:max_col] = bin[min_row:max_row, min_col:max_col]

    except Exception as e:
        logging.error("error in square building")
        raise e

    # close blanks
    bw = closing(blob_bin, square(2))

    # label image regions
    label_image = label(bw)

    return label_image, blob_bin


def label_contours(contours, shape, close_square = 2):
    binary = np.zeros(shape)
    for c in contours:
        rr, cc = polygon(c[:, 0], c[:, 1], binary.shape)
        binary[rr, cc] = 1
        # close blanks

    #bw = closing(binary, square(close_square))

    # label image regions
    label_image = label(binary)
    return label_image, binary

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


def overlaped_regions(im1, df_region1, prev_im, prev_regions, filter=0.1):
    """
    Identify overlaped regions between two images. Return a list of tuple containing (region from img 1, region from img 2)
    :param im1: (N, M) ndarray
        First image to compare
    :param df_region1: list<RegionProperties>
        RegionProperties extracted from im1
    :param prev_im: (N, M) ndarray
        previous image to compare with
    :param prev_regions: list<RegionProperties>
        RegionProperties extracted from previous image
    :param filter: float, int
        Value of threshold used to generate binary image to compare
        Default : 100
    :return: region_couples : dict<(int, RegionProperties)>
        The regions that overlap from im1 to im2 mapped with their Region3D_id
    :return: new_regions_matched_ids: list<int>
        Ids of regions from the new image that matched with existing Region3D
    """

    if isinstance(prev_regions, dict):
        prev_regions = prev_regions
    elif isinstance(prev_regions, list):
        prev_regions = {key: value for key, value in enumerate(prev_regions)}
    elif isinstance(prev_regions, pd.DataFrame):
        prev_regions = {key: value for key, value in prev_regions.iterrows()}
    else:
        raise TypeError("Wrong type of values for regions2")

    # Compute difference between the two images
    bin1 = label_filter(im1, filter)[1]
    bin2 = label_filter(prev_im, filter)[1]
    overlap_bin = np.logical_and(bin1, bin2)

    label_overlap_image = label(overlap_bin)
    df_overlap_prop = region_properties(label_overlap_image, properties=['centroid'])
    centroids = [(region['centroid-0'], region['centroid-1']) for i, region in df_overlap_prop.iterrows()]

    def build_regions_table(regions):
        """
        Build dictionnary to find region id thanks to coordinate research
        :param regions: List of regions
        :return: region_table: Dictionnary
        """
        if isinstance(regions, pd.DataFrame):
            regions = regions.iterrows()
        elif isinstance(regions, dict):
            regions = regions.items()
        else:
            raise TypeError("Wrong type of region callable. Use list or dict")

        regions_table = {}
        for idx, region in regions:
            for coord in region['coords']:
                regions_table[(coord[0], coord[1])] = idx
        return regions_table

    regions1_table = build_regions_table(df_region1)
    prev_regions_table = build_regions_table(prev_regions)

    existing_regions_map = {}
    new_regions_matched_ids = []
    logging.info("start mapping")
    matching_fail = 0
    for centroid in centroids:
        try:
            existing_regions_map[prev_regions_table[centroid]] = df_region1.iloc[regions1_table[centroid]]
            new_regions_matched_ids.append(regions1_table[centroid])
        except KeyError as e:
            logging.debug("Centroid unmatched {}".format(centroid))
            matching_fail += 1
    logging.info("Amount of overlaped regions :" + str(len(centroids)))
    logging.info("Amount of regions mapped : " + str(len(existing_regions_map)))
    if len(centroids) > 0:
        logging.info("Matching fails ratio :" + str(round(matching_fail / len(centroids) * 100)) + "%")
    return existing_regions_map, new_regions_matched_ids


# ________________________________________________
#              CLASSIFICATION
# ________________________________________________


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
