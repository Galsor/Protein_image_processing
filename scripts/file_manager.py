import datetime
import logging
import pickle
import time

import pandas as pd
from skimage import io
import sys
import os
from ast import literal_eval

FILE_NAME_TEMPLATE = "C10DsRedlessxYw_emb{}_Center_Out.tif"
# DATA_PATH = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data"

PATH_TO_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PATH_TO_ROOT_DIR = os.path.normpath(os.path.join(PATH_TO_CURRENT_DIR, '..'))
sys.path.append(PATH_TO_ROOT_DIR)

DATA_PATH = os.path.join(PATH_TO_ROOT_DIR, 'data')

EMBRYOS = [1, 7, 8, 10, 11]


def get_embryos():
    return EMBRYOS


def get_tiff_file(embryo):
    file_path = DATA_PATH + "\\embryos\\" + FILE_NAME_TEMPLATE.format(embryo)
    tiff = io.imread(file_path)
    return tiff


def save_results(df, file_name=None, timestamped=False):
    directory = DATA_PATH + "\\results\\"
    file_name = format_file_name(file_name, directory, timestamped, ".csv")
    if isinstance(df, pd.DataFrame):
        df.to_csv(file_name)
    else:
        raise TypeError("Inputs of save_results are not pandas.DataFrame")


def get_test_features(file_name="Features_cells_C10DsRedlessxYw_emb11_Center_Out.tif.csv"):
    if file_name[:-4]!=".csv":
        file_name+=".csv"

    #path = os.path.join(DATA_PATH, file_name)
    path = exist(file_name)
    try:
        df = pd.read_csv(path)
    except:
        raise ValueError(" '{}' couldn't be found in data directory".format(file_name))
    return df


def save_timestamped_results(df, title):
    file_name = title + "{}".format(datetime.now()) + ".csv"
    save_results(df, file_name=file_name)


def save_as_pickle(data, file_name=None, timestamped=False):
    directory = DATA_PATH + "\\variables\\"
    file_name = format_file_name(file_name, directory, timestamped, ".pkl")
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print("An error occured while saving {}".format(file_name))
        print(repr(e))


def load_pickle(file_name):
    directory = DATA_PATH + "\\variables\\"
    file_name = directory + file_name
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def format_file_name(file_name, directory, timestamped, extension):
    if file_name is not None and file_name[-4:] == extension:
        # remove extension
        file_name = file_name[:-4]

    if file_name is None:
        file_name = directory + "Results_{}".format(time.strftime("%Y-%m-%d-%H%M")) + extension
    elif timestamped:
        file_name = directory + file_name + "_" + time.strftime("%Y-%m-%d-%H%M") + extension
    else:
        file_name = directory + file_name + extension
    return file_name


def exist(file_name):
    paths = [DATA_PATH]
    for dir in get_data_subdirs():
        paths.append(DATA_PATH+"\\"+dir)
    for path in paths:
        if os.path.exists(path+"\\"+file_name) :
            return path+"\\"+file_name
    return False


def get_data_subdirs():
    dirs = []
    for x in os.listdir(DATA_PATH):
        if os.path.isdir(DATA_PATH+"\\"+x):
            dirs.append(x)
    return dirs