"""
file_processing contains functions allowing to extract cells and proteins from the data of a tiff file.
Theses functions wraped lower level functions of image processing and regions extractions and returns RegionFrame object containing a list of Region3D.
This RegionFrame can be used for the features extraction needed to achieve classification.
"""


import logging

import scripts.file_manager as fm
from scripts.preprocessing import blob_extraction, label_filter, label_filter_blobs, find_cells_contours, \
    label_filter_contours, define_filter_value
from scripts.region import region_properties, overlapped_regions
from scripts.region_3d import RegionFrame



PROPERTIES = ['area',
              'centroid',
              'coords',
              'extent',
              'max_intensity',
              'mean_intensity',
              'min_intensity',
              'intensity_image',
              'convex_area']


def extract_regions(tiff, channel=0, min_area=2, filter="auto", cells=None, mode='region'):
    """
    Extract regions3D from each layers of a tiff file.

    :param tiff: ndarray
        4 dimensions array of images in format (layers, x_img, y_img, channels)
        ex: (40, 1000, 1000, 3)
    :param channel: int
        Channel to use for region extraction.
        For mollecules use 0, for cells use 2.
    :param min_area: int, float
        Minimum value of region area size.
        Increase the value of min_area to reject noise.
    :param filter: int, float, tuple<int>, str.
        Filter value. Refer to label_filter for further information
    :param mode: str
        Type of region detection used. 2 modes are available : "region" and "blob".
    :param cells: ndarray, optional
        Image including labels of cells.
    :return: rf: RegionFrame
        RegionFrame object containing all regions 3D extracted in the image.
    """
    if not isinstance(channel, int):
        raise TypeError("Wrong type for channel value")

    try:
        ch = tiff[:, :, :, channel]
    except Exception as e:
        raise e

    # Workaround to remove after refactoring of label filtering
    if filter == "auto":
        filter = define_filter_value(ch, filter = filter)

    if mode == 'blob':
        #Todo : refactor and create a step safe mode with pickle saving. Clean pickles after complete processing of the pipeline.
        if fm.exist("data_test_blob_detect.pkl"):
            blobs, rscl_img = fm.load_pickle("data_test_blob_detect.pkl")
        else:
            logging.info(" === BLOB DETECTION ===")
            blobs, rscl_img = blob_extraction(ch)

    init = True
    logging.info("=== REGION EXTRACTION ===")
    for layer, img in enumerate(ch):
        logging.info("_" * 80)
        logging.info("Layer {}".format(layer))
        logging.info("_" * 80)
        logging.info("Extract regions")
        if mode == 'region':
            df_properties = region_properties(label_filter(img, filter=filter)[0], img, properties=PROPERTIES,
                                              min_area=min_area, cells=cells)
        if mode == 'blob':
            #img = rscl_img[layer]
            layer_blobs = blobs[layer]
            df_properties = region_properties(label_filter_blobs(img, layer_blobs, filter=filter)[0], img, properties=PROPERTIES,
                                              min_area=min_area, cells=cells)
        # Previously test if 'regions' existing. Region has been replaced by df_properties
        logging.info("Add results to the RegionFrame")
        if init:
            rf = RegionFrame(df_properties)
            init = False
            prev_img = img
        elif not init:
            region_dict = rf.get_regions_in_last_layer()
            logging.info("Look for overlapped regions")
            matched_regions, new_regions = overlapped_regions(img, df_properties, prev_img, region_dict,
                                                              filter=filter)
            existing_regions_map = rf.enrich_region3D(matched_regions)
            if not new_regions.empty:
                new_regions_map = rf.populate_region3D(new_regions)
            else:
                new_regions_map = []
            rf.update_map(existing_regions_map, new_regions_map)
            prev_img = img

    logging.info("Total amount of regions detected {}".format(rf.get_amount_of_regions3D()))
    return rf


def extract_cells(tiff):
    """Generate labeled image of each cells in the tiff file.
    This methods uses contours detection to isolate cells in channel 2 of the tiff file.

    :param tiff: ndarray
        4 dimensions array of images in format (layers, x_img, y_img, channels)
        ex: (40, 1000, 1000, 3)
    :return: label_img: ndarray<int>
        single image labelazing each cells detected.
    """
    #TODO : Remove the overlabelling cases (where several cells are labelled as one.
    ch3 = tiff[:,:,:,2]
    contours = find_cells_contours(ch3)
    label_img, bin = label_filter_contours(contours, ch3[0].shape)
    return label_img


def extract_region_with_cells(tiff, channel=0, min_area=2, filter="auto", mode='region'):
    label_cells = extract_cells(tiff)
    rf = extract_regions(tiff, cells=label_cells, channel=channel, min_area=min_area, filter=filter, mode=mode)
    return rf


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    embs = fm.get_embryos()
    for emb in embs :
        tiff = fm.get_tiff_file(emb)
        rf = extract_region_with_cells(tiff)
        df = rf.extract_features()
        print("Regions detected : {}".format(len(df)))
        fm.save_results(df, file_name="features_extraction_auto_emb{}_".format(emb), timestamped=True)