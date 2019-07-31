import pandas as pd
from skimage import io, color, measure
import matplotlib.pyplot as plt
import numpy as np
from Multi_slice_viewer import display_file, multi_slice_viewer, ax_config, process_key
import os
import sys

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

# Import for K-means clustering
from sklearn.cluster import KMeans

"""
OBJ : Compter le nombre de TS et single mol par noyau
Règle : Max 2 TS par noyau

"""

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


def spectrum_viewer(im):
    red, green, blue = extract_channels(im)

    im_hsv = convert_HSV(im)
    hue, saturation, value = extract_channels(im_hsv)

    im_hed = convert_HED(im)
    haematoxylin, eosin, dab = extract_channels(im_hed)

    ims = [red, green, blue, hue, saturation, value, haematoxylin, eosin, dab]
    # shape = (9,) + im.shape
    # ims = np.empty(shape)
    # ims = ims red

    fig, axes = plt.subplots(3, 3, figsize=(7, 6), sharex=True, sharey=True)
    labels = ["red", "green", "blue", "Hematoxylin", "Eosin", "DAB", "Hue", "Saturation", "Value"]
    ax = axes.ravel()

    for i in np.arange(len(labels)):
        ax_config(ims[i], ax[i])
        ax[i].set_title(labels[i])

    fig.canvas.mpl_connect('key_press_event', process_key)
    fig.tight_layout()
    plt.show()


def plot_img(image, cmap='gnuplot2', title="Undefined title"):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    fig.suptitle(title)
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

def simple_filter(image):
    np.place(image, image < 100, 0)
    from skimage.measure import regionprops
    props = regionprops(image)
    print(props)


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
    print("# of regions : " + str(len(regionprops(label_maxima))))
    if h:
        h_maxima = extrema.h_maxima(img, h)
        label_h_maxima = label(h_maxima)
        overlay_h = color.label2rgb(label_h_maxima, img, alpha=0.7, bg_label=0,
                                    bg_color=None, colors=[(1, 0, 0)])
        row, col = add_img(overlay_h, axs, col=col, row=row, title="local maximas thresholded")
        print("# of regions after thresholding : " + str(len(regionprops(label_h_maxima))))
    return img, label_maxima, label_h_maxima


# todo : try hysteris detection : https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html#sphx-glr-auto-examples-filters-plot-hysteresis-py

# todo : Try entropy to enhace edges and facilitate detection https://scikit-image.org/docs/dev/auto_examples/filters/plot_entropy.html#sphx-glr-auto-examples-filters-plot-entropy-py

# todo : try template matching https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_template.html#sphx-glr-auto-examples-features-detection-plot-template-py

# todo : [Top] Try blob detection : https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html#sphx-glr-auto-examples-features-detection-plot-blob-py

# todo : Try ridges operator to enhance edges : https://scikit-image.org/docs/dev/auto_examples/edges/plot_ridge_filter.html#sphx-glr-auto-examples-edges-plot-ridge-filter-py

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
            raise repr(err)

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
            print("Error with : " + prop)
            print(repr(err))
    r_properties = pd.DataFrame(r_properties)
    return regions, r_properties


def demo_regions(image, label_image):
    fig, axs = plt.subplots(ncols=2, figsize=(10, 6))
    axs[0].imshow(image, cmap='gnuplot2')

    # Compute regions properties
    regions, props = region_properties(label_image, image, min_area=4,
                                       properties=['extent', 'max_intensity', 'area', "mean_intensity"])
    intensity_features = []
    for region in regions:
        # take regions with large enough areas
        if region.area >= 4:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            axs[0].add_patch(rect)
            intensity_features.append(extract_intensity_features_from_region(region, image))

    print(props.shape)
    print(props.head())
    print(intensity_features[0:5])

    sns.scatterplot(size=props['extent'], hue=props['max_intensity'], x=props['area'], y=props["mean_intensity"],
                    ax=axs[1])
    axs[0].set_axis_off()
    plt.tight_layout()

    return regions, props


def label_filter(image, filter=None, window_size=5, k=0.2):
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

    # Apply filter

    if not filter:
        thresh = 100
    elif isinstance(filter, int):
        thresh = filter
    elif filter == "otsu":
        thresh = threshold_otsu(image)
    elif filter == "niblack":
        thresh = threshold_niblack(image, window_size=window_size, k=k)
        # raise "Not implemented yet"
    elif filter == "sauvola":
        thresh = threshold_sauvola(image, window_size=window_size)
        # raise "Not implemented yet"
    print("Threshold : ")
    print(thresh)

    binary = image > thresh

    bw = closing(binary, square(2))

    # label image regions
    label_image = label(bw)
    image_label_overlay = label2rgb(label_image, image=image)

    return label_image, image_label_overlay, binary


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

    print("Number of matches:", matches.shape[0])
    print("Number of inliers:", inliers.sum())

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


# todo : integrate the function in 3D layers analyzer
def overlaped_regions(im1, regions1, im2, regions2, threshold=100):
    """
    Identify overlaped regions between two images. Return a list of tuple containing (region from img 1, region from img 2)
    :param im1: (N, M) ndarray
        First image to compare
    :param regions1: list<RegionProperties>
        RegionProperties extracted from im1
    :param im2: (N, M) ndarray
        Second image to compare
    :param regions2: list<RegionProperties>
        RegionProperties extracted from im2
    :param threshold: float, int
        Value of threshold used to generate binary image to compare
        Default : 100
    :return: region_couples : list<(RegionProperties, RegionProperties)>
        The pairs of regions that overlap from im1 to im2.
    """
    bin1 = im1 > threshold
    bin2 = im2 > threshold
    overlap_bin = np.logical_and(bin1, bin2)
    label_overlap_image = label(overlap_bin)
    overlap_regions, df_overlap_prop = region_properties(label_overlap_image)
    centroids = [list(region.centroid) for region in overlap_regions]
    centroids = [(round(centroid[0]).astype('Int64'), round(centroid[1]).astype('Int64')) for centroid in centroids]

    def build_regions_table(regions):
        """
        Build dictionnary to find region id thanks to coordinate research
        :param regions: List of regions
        :return: region_table: Dictionnary
        """
        regions_table = {}
        for idx, region in enumerate(regions1):
            for coord in region.coords:
                regions_table[(coord[0], coord[1])] = idx
        return regions_table

    regions1_table = build_regions_table(regions1)
    regions2_table = build_regions_table(regions2)

    region_couples = []
    print("start mapping")
    fails = 0
    for centroid in centroids:
        try:
            region_couples.append((regions1[regions1_table[centroid]], regions2[regions2_table[centroid]]))
        except Exception as e:
            fails += 1
            pass
    print(len(centroids))
    print(len(region_couples))
    print(fails)
    return region_couples


# ________________________________________________
#              CLASSIFICATION
# ________________________________________________
def kmeans_classification(features, n_clusters=2, tol=1e-5):
    k_means = KMeans(n_clusters=n_clusters, n_init=1000, max_iter=10000)
    k_means.fit(features)
    labels = pd.Series(k_means.labels_)
    # print(labels.to_string())
    # TS = labels.where(labels==1)
    # protein = labels.where(labels==0)
    return k_means, labels


def plot_result_classif(regions, properties, labels, image):
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
    file_path = os.path.join(DATA_PATH, FILE_NAME)
    im = io.imread(file_path)
    ch1 = im[:, :, :, 0]
    image = ch1[14]
    next = ch1[15]

    regions_14, df_props_14 = region_properties(label_filter(image)[0], image, min_area=4)
    regions_15, df_props_15 = region_properties(label_filter(next)[0], next, min_area=4)

    overlaped_regions(image, regions_14, next, regions_15)

    # plot_img(image, title="Channel 1, z=14")
    # label_image, image_label_overlay, binary = label_filter(ch1[14])
    # regions, df_props = region_properties(label_image, image, min_area=4, properties=['extent','max_intensity','area',"mean_intensity"])

    # if col.dtype is object or col.isnull() == True then drop column
    # features = df_props.drop(["coords","convex_image","filled_image","image","intensity_image", "slice", "moments_normalized-0-0","moments_normalized-0-1", "moments_normalized-1-0","weighted_moments_normalized-0-0","weighted_moments_normalized-0-1", "weighted_moments_normalized-1-0"], axis=1)
    # print(features.isnull().any().to_string())

    # kmeans, labels = kmeans_classification(df_props)
    # plot_result_classif(regions,df_props,labels,label_image)

    plt.show()
    # hough_circle_detection(test[10])

    # img_10, label_maxima_10, label_h_maxima_10 = local_maximas(test[10], h=0.05)
    # img_11, label_maxima_11, label_h_maxima_11 = local_maximas(test[11], h=0.05)

    # hough_elliptic_detection(test[10])

    # with tifffile.TiffFile(file_path) as tif:
    #     data = tif.asarray()
    #     print("Pages")
    #     print(len(tif.pages))
    #     print("Series")
    #     print(len(tif.series))
    #     print("is flagged")
    #     #print(tif.is_flag())
    #     print("to string")
    #     print(str(tif))
    #     page = tif.pages[50]
    #     print("tags")
    #     print(page.tags.keys())
    #     print("photometric")
    #     print(page.photometric)
    #     #rgb = page.asrgb()
    # #io.imshow(red[14])
    # #multi_slice_viewer(red)
    # #tifffile.imshow(red, title="RGB", cmap='gnuplot')
    # #tifffile.imshow(data, photometric='MINISBLACK',cmap='gnuplot', title='MINISBLACK')
    # print(data.shape)

    # label_image_region(im[0])
    # display_file(file_path)
