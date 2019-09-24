""" This file implements a complete application of the pipeline :
- File openning
- Image processing (Regions bounding)
- Features extractions
- Classification (using MeanShift)
- Result ploting
"""
import logging

import scripts.file_manager as fm
from scripts.classifiers import classify
from scripts.file_processing import extract_region_with_cells
from scripts.file_viewer import MultiLayerViewer

EMBRYO = 1
#TODO : add logging status
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # get a file from its embryo code
    tiff = fm.get_tiff_file(EMBRYO)
    # Extract a RegionFrame including all the regions 3D of the image
    rf = extract_region_with_cells(tiff, filter=100)
    # Extract features from the regions
    features = rf.extract_features()
    # Classify features using tail filtering, PCA and MeanShift
    f_label = classify(features)
    #Plot results
    viewer = MultiLayerViewer(tiff)
    viewer.plot_imgs(features=f_label)
    viewer.show()
