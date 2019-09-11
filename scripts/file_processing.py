"""
file_processing contains functions allowing to extract cells and proteins from the data of a tiff file.
Theses functions wraped lower level functions of image processing and regions extractions and returns RegionFrame object containing a list of Region3D.
This RegionFrame can be used for the features extraction needed to achieve classification.
"""


import logging

import scripts.file_manager as fm
from scripts.preprocessing import blob_extraction, label_filter, label_filter_blobs, find_cells_contours, \
    label_filter_contours
from scripts.region import region_properties, overlaped_regions
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


def extract_regions(tiff, channel=0, min_area=2, filter=0.10, cells=None, mode='region'):
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
    if mode == 'blob':
        if fm.exist("data_test_blob_detect.pkl"):
            blobs, rscl_img = fm.load_pickle("data_test_blob_detect.pkl")
        else:
            blobs, rscl_img = blob_extraction(ch)

    init = True
    for layer, img in enumerate(ch):
        logging.info("_" * 80)
        logging.info("Layer {}".format(layer))
        logging.info("_" * 80)
        if mode == 'region':
            df_properties = region_properties(label_filter(img, filter=filter)[0], img, properties=PROPERTIES,
                                              min_area=min_area, cells=cells)
        if mode == 'blob':
            #img = rscl_img[layer]
            layer_blobs = blobs[layer]
            df_properties = region_properties(label_filter_blobs(img, layer_blobs, filter=filter)[0], img, properties=PROPERTIES,
                                              min_area=min_area, cells=cells)
        # Previously test if 'regions' existing. Region has been replaced by df_properties
        if init:
            rf = RegionFrame(df_properties)
            init = False
            prev_img = img
        elif not init:
            region_dict = rf.get_regions_in_last_layer()
            matched_regions, new_regions = overlaped_regions(img, df_properties, prev_img, region_dict,
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

def extract_region_with_cells(tiff, channel=0, min_area=2, filter=0.10, mode='region'):
    label_cells = extract_cells(tiff)
    rf = extract_regions(tiff, cells=label_cells, channel=channel, min_area=min_area, filter=filter, mode=mode)
    return rf



def test_pipeline():
    # function
    file_path_template = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data\\embryos\\" \
                         "C10DsRedlessxYw_emb{}_Center_Out.tif"
    EMBRYOS = {1: (77, 24221), 7: (82, 23002), 8: (71, 15262), 10: (92, 23074)}

    try:
        results = {}
        for embryo in EMBRYOS.keys():
            file_path = file_path_template.format(embryo)
            results[embryo] = extract_regions(file_path).get_amount_of_regions3D()
        for embryo, r in results.items():
            expected = EMBRYOS[embryo][0] + EMBRYOS[embryo][1]
            logging.info(" {} (expected {} ) proteins detected for embryo {}").format(r, expected, embryo)

    except Exception as e:
        raise e

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    embs = fm.get_embryos()
    for emb in embs :
        tiff = fm.get_tiff_file(emb)
        rf = extract_region_with_cells(tiff,filter = 100)
        df = rf.extract_features()
        fm.save_results(df, file_name="features_extraction_emb{}_".format(emb), timestamped=True)