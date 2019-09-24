import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
import matplotlib.patches as mpatches
from skimage.measure import regionprops, label

from scripts.preprocessing import label_filter


# ________________________________________________
#         REGIONS PROPERTIES EXTRACTION
# ________________________________________________

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


def region_properties(label_image, image=None, min_area=1, max_area=0.002, properties=None, cells=None, separator='-'):
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
    image : (N, M) ndarray, optional
        Intensity (i.e., input) image with same size as labeled image.
        Default is None.
    min_area : int or float optional
        Minimum area size of regions. if < 1, a percentage of the image size will be used as min_area.
        Default is 1.
    min_area : int or float optional
        Maximum area size of regions. if < 1, a percentage of the image size will be used as max_area.
        Default is 0.002 (i.e : 0.2% of image total size).
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
    r_properties : pandas.DataFrame
        DataFrame with rows as regions and columns as region properties`.

    """

    if image is None:
        regions = regionprops(label_image)
    else:
        try:
            regions = regionprops(label_image, image)
        except Exception as err:
            raise err

    passband = [min_area, max_area]
    for i, thresh in enumerate(passband):
        if thresh < 1:
            passband[i] = label_image.shape[0]*label_image.shape[1]*thresh
    # Select only the regions up to the min area criteria
    regions = [region for region in regions if region.bbox_area > passband[0] and region.bbox_area < passband[1]]

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
    r_properties = add_cells(r_properties, cells)
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
#               OVERLAPED REGION FINDER
# ________________________________________________

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
    :return: existing_regions_map : dict<(int, RegionProperties)>
        Map of regions overlaping from im1 to im2 with their related Region3D_id
    :return: regions_unmatched: pandas.DataFrame
        Dataframe of regions that didn't match with existing Region3D
    """
    # format prev_regions to a dictionary dtype.
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

    # label the region where an overlap appears
    label_overlap_image = label(overlap_bin)

    # extract the properties of the overlapped regions
    df_overlap_prop = region_properties(label_overlap_image, properties=['centroid'])
    # collect the centroids of the overlapped regions
    centroids = [(region['centroid-0'], region['centroid-1']) for i, region in df_overlap_prop.iterrows()]

    def build_regions_table(regions):
        """ Map all regions coordinates with their related region id

        :param regions: List of regions
        :return: region_table: dict
            Keys: tuple of coordinate (x,y)
            Values: region id
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
            #Try to find a new region that includes the centroid of the overlap region
            existing_regions_map[prev_regions_table[centroid]] = df_region1.iloc[regions1_table[centroid]]
            new_regions_matched_ids.append(regions1_table[centroid])
        except KeyError as e:
            logging.debug("Centroid unmatched {}".format(centroid))
            matching_fail += 1

    # create the list of the regions that have not match with an overlap centroid
    regions_unmatched = df_region1.drop(new_regions_matched_ids)

    logging.info("Amount of overlaped regions :" + str(len(centroids)))
    logging.info("Amount of regions mapped : " + str(len(existing_regions_map)))
    if len(centroids) > 0:
        logging.info("Matching fails ratio :" + str(round(matching_fail / len(centroids) * 100)) + "%")
    return existing_regions_map, regions_unmatched

def add_cells(r_properties, label_cells):
    """Check if the centroids of the detected regions are included in a cell. If yes, add the cell label to the region properties """
    if label_cells is not None:
        cells =[label_cells[r['centroid-0']][r['centroid-1']] for i, r in r_properties.iterrows()]
        cells = pd.Series(cells, index=r_properties.index)
        cells.name = "cell"
        r_properties = pd.concat([r_properties,cells], axis=1)
    return r_properties

