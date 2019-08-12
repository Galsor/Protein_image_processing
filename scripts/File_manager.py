import datetime
import pandas as pd
from skimage import io
import sys
import os

FILE_NAME_TEMPLATE = "C10DsRedlessxYw_emb{}_Center_Out.tif"
#DATA_PATH = "C:\\Users\\Antoine\\PycharmProjects\\Protein_image_processing\\data"

PATH_TO_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PATH_TO_ROOT_DIR = os.path.normpath(os.path.join(PATH_TO_CURRENT_DIR, '..'))
sys.path.append(PATH_TO_ROOT_DIR)

DATA_PATH = os.path.join(PATH_TO_ROOT_DIR, 'data')

EMBRYOS = [1,7,8,10,11]

def get_embryos():
    return EMBRYOS

def get_tiff_file(embryo):
    file_path = DATA_PATH + "\\embryos\\" + FILE_NAME_TEMPLATE.format(embryo)
    tiff = io.imread(file_path)
    return tiff

def save_results(df, file_name = None):
    directory = DATA_PATH+"\\results\\"
    if file_name is None:
        file_name = directory+"Results_{}".format(datetime.now())
    else :
        file_name = directory + file_name
    if isinstance(df, pd.DataFrame):
        df.to_csv(file_name)
    else :
        raise TypeError("Inputs of save_results are not pandas.DataFrame")

def save_timestamped_results(df, title):
    file_name = title+"{}".format(datetime.now())+".csv"
    save_results(df, file_name = file_name)