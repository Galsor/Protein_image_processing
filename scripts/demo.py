""" This file implements a complete application of the pipeline :
- File openning
- Image processing (Regions bounding)
- Features extractions
- Classification (using MeanShift)
- Result ploting
"""
import logging
import os

import scripts.file_manager as fm
from scripts.classifiers import classify, save_results_classif
from scripts.file_processing import extract_region_with_cells
from scripts.file_viewer import MultiLayerViewer

EMBRYO = 1
#TODO : Save results with all features labelled
#TODO : Add input to paste file directory

def get_path_input():
    path = input('Please enter the complete path of the tiff file you want to analyse: ')
    if os.path.isfile(path):
        return path
    else:
        print("The path you entered is not a file")
        return get_path_input()

def get_res_file_name():
    print("Data processing is now finished")
    print("I am ready to save the results.")
    file_name = input('''Which file name do you want to use (press enter to use "classification_results") ?''')
    return file_name


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Hello my friend,")
    # get a file from its embryo code
    #tiff = fm.get_tiff_file(EMBRYO)
    #get a file from path
    path = get_path_input()
    tiff = fm.get_tiff_file(path)
    # Extract a RegionFrame including all the regions 3D of the image
    rf = extract_region_with_cells(tiff, filter=100)
    # Extract features from the regions
    features = rf.extract_features()
    # Classify features using tail filtering, PCA and MeanShift
    f_label = classify(features)
    file_name = get_res_file_name()
    if file_name == "":
        file_name = "classification_results"
    save_results_classif(features, f_label, file_name=file_name)

    #Plot results
    viewer = MultiLayerViewer(tiff)
    viewer.plot_imgs(features=f_label)
    viewer.show()
