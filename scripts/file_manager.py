""" This file supply a bunch of tools to facilitate file manaement, especially for file included in the ./data folder.
"""

import pickle
import time

import pandas as pd
from skimage import io
import sys
import os


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
    """ Open a tiff file from the id of the embryo.

    :param embryo: int
        Key or id used to identify the embryo in the file name. Template use for file naming must be : C10DsRedlessxYw_emb{embryo id}_Center_Out.tif
    :return: tiff: ndarray
        4 dimensional array including all images and all channels
    """
    file_path = DATA_PATH + "\\embryos\\" + FILE_NAME_TEMPLATE.format(embryo)
    tiff = io.imread(file_path)
    return tiff


def save_results(df, file_name=None, timestamped=False):
    """ Save pandas.DataFrame results in csv file.

    :param df: pandas.DataFrame
        Input DataFrame to export in .csv format
    :param file_name: str
        Name of the file
    :param timestamped: bool
        If yes, add a timestamp at the end of the file name
    """

    directory = DATA_PATH + "\\results\\"
    file_name = format_file_name(file_name, directory, timestamped, ".csv")
    if isinstance(df, pd.DataFrame):
        df.to_csv(file_name)
    else:
        raise TypeError("Inputs of save_results are not pandas.DataFrame")


def get_data_from_file(file_name="Features_cells_C10DsRedlessxYw_emb11_Center_Out.tif.csv"):
    """ Return DataFrame from existing .csv file in the data directory.

    :param file_name: str
        file name. Can be setted with or without the csv extention.
    :return: df: pandas.DataFrame
        the data.
    """
    # TODO Explore all subdirectories of ./data (even sub-sub directories)
    if file_name[-4:]!=".csv":
        file_name+=".csv"

    #path = os.path.join(DATA_PATH, file_name)
    path = exist(file_name)
    try:
        df = pd.read_csv(path)
    except:
        raise ValueError(" '{}' couldn't be found in data directory".format(file_name))
    return df


def save_as_pickle(data, file_name=None, timestamped=False):
    """ Save python object in pickle format. Object can be of any pythonic type. Pickle file will be saved in ./data/variables directory

    :param data: Object
        Data to save
    :param file_name: str
        file name
    :param timestamped: bool
        If yes, add a timestamp at the end of the file name
    """
    directory = DATA_PATH + "\\variables\\"
    file_name = format_file_name(file_name, directory, timestamped, ".pkl")
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print("An error occured while saving {}".format(file_name))
        print(repr(e))


def load_pickle(file_name):
    """ Load pickle file from ./data/variables directiory.

    :param file_name:
        Name of the file
    :return: data: Object
        Python object stored in the pickle file
    """
    directory = DATA_PATH + "\\variables\\"
    file_name = format_file_name(file_name, directory, False, '.pkl')
    file_name = directory + file_name
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data


def format_file_name(file_name, directory, timestamped, extension):
    """ Util that facilitate file name management by generating a standard and complete file name from the input conditions.

    :param file_name: str
        Input file name
    :param directory: str
        Directory where the file is stored
    :param timestamped: bool
        If yes, add timestamp at the end of the file name and prior to its extension
    :param extension: str
        Extention of the file. Preferably '.csv' or '.pkl'.
    :return: file_name: str
        Formated file name.
    """

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
    """ Check if a file exist in any of the subdirectories of ./data

    :param file_name: str
        File name
    :return:
        if file exist: str
            Complete path to access the file.
        else : bool
            False
    """
    paths = [DATA_PATH]
    for dir in get_data_subdirs():
        paths.append(DATA_PATH+"\\"+dir)
    for path in paths:
        if os.path.exists(path+"\\"+file_name) :
            return path+"\\"+file_name
    return False


def get_data_subdirs():
    """ Returns all subdirectories of ./data
    :return: array
        list of directories
    """
    dirs = []
    for x in os.listdir(DATA_PATH):
        if os.path.isdir(DATA_PATH+"\\"+x):
            dirs.append(x)
    return dirs


def get_files(dir = None):
    files = []
    dir_path = DATA_PATH
    if dir is not None :
        dir_path = os.path.join(DATA_PATH, dir)

    for x in os.listdir(dir_path):
        if os.path.isfile(dir_path + "\\" + x):
            files.append(x)
    return files