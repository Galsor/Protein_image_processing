import logging
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.signal import find_peaks

from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola
from skimage.morphology import closing, square
from skimage.measure import label
from skimage.feature import blob_log
from skimage.exposure import rescale_intensity, histogram
from skimage.draw import polygon
from skimage import measure

import scripts.file_manager as fm


# ________________________________________________
#               CONTOURS DETECTION
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


# ________________________________________________
#               INTENSITY PEAKS DETECTION
# ________________________________________________

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


# ________________________________________________
#               BLOB DETECTION
# ________________________________________________

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
#               IMAGE RESCALING
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


# ________________________________________________
#               IMAGE LABELLING
# ________________________________________________

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


def label_filter_blobs(img, blobs, filter=None, window_size=5, k=0.2):
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


def label_filter_contours(contours, shape, close_square = 2):
    binary = np.zeros(shape)
    for c in contours:
        rr, cc = polygon(c[:, 0], c[:, 1], binary.shape)
        binary[rr, cc] = 1
        # close blanks

    #bw = closing(binary, square(close_square))

    # label image regions
    label_image = label(binary)
    return label_image, binary