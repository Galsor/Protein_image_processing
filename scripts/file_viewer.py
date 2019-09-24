"""
file_viewer contains several functions allowing to display multi-layer images such as tiff files.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from math import sqrt
from skimage import io
import numpy as np
import pandas as pd

import scripts.file_manager as fm
from scripts.preprocessing import label_filter
from scripts.region import region_properties

CONFIG = ['standard', 'blob', 'region','classif']

class MultiLayerViewer():
    """Matplotlib viewer for diaporama display. This viewer has been configured to fit with miscroscope pictures (i.e several images 1000pxx1000px containing multiple channels.

    Parameters
    ----------
    tiff : ndarray<int>
        4 dimensional array containing all the images and channels. This input can be the one directly extracted from the tiff file.

    config : str
        Configuration used for displaying. 2 modes are available :
            - 'standard' : simple display of images
            - 'blob' : Add patches representing blobs in overlay of the standard display.
            - 'region' : Add rectangles representing bbox of regions.
            - 'classif': Add colored square patches representing the result of the classification

    channel : int
        Index of the channel to display.
        Default : 0
    """
    def __init__(self, tiff, config = CONFIG[0], channel = 0):
        #TODO : Add figure title
        self.set_config(config)
        self.config = config
        self.channel = channel
        if isinstance(tiff, str):
            tiff = io.imread(tiff)
            try :
                self.imgs = tiff[:,:,:,channel]
            except ValueError as e :
                raise e
            except Exception as e :
                raise e
        elif isinstance(tiff, np.ndarray):
            if len(tiff.shape) == 2:
                self.imgs = np.array(tiff)
            elif len(tiff.shape) == 3:
                self.imgs = tiff
            elif len(tiff.shape) == 4:
                self.imgs = tiff[:,:,:,channel]
            else :
                raise Exception("Some error occures while checking tiff file length.")

    def set_config(self, config):
        if config not in CONFIG:
            raise ValueError('Unvalid config value for multi_slice_viewer. Please use {}'.format(CONFIG))
        else :
            self.config = config

    def get_images(self):
        return self.imgs

    def set_images(self, tiff, channel=None):
        if isinstance(tiff, np.ndarray):
            if len(tiff.shape) == 2:
                self.imgs = np.array(tiff)
            elif len(tiff.shape) == 3:
                self.imgs = tiff
            elif len(tiff.shape) == 4 and channel is not None :
                self.imgs = tiff[:, :, :, channel]
        else :
            raise Exception("Impossible to set new images for MultiSliceViewer")

    def plot_imgs(self, blobs = None, properties = None, features = None):
        """ Configure the matplotlib Figure and Axes. Call this method to plot images included in the MultiLayerViewer.
        call MultiLayerViewer.show() to display the resulting plot.

        :param blobs: array
            List of parameters resulting from preprocessing.blob_extraction method.
        """

        #Plot preparation
        if len(self.imgs.shape) == 2:
            # convert image into a list of images.
            self.imgs = np.array(self.imgs)
        self.remove_keymap_conflicts({'j', 'k'})
        if blobs is not None:
            # Change configuration is blobs are passed as inputs
            self.set_config(CONFIG[1])
        elif properties is not None:
            self.set_config(CONFIG[2])
        elif features is not None:
            self.set_config(CONFIG[3])
        else :
            # If no blobs are passed as input, go to standard configuration
            self.set_config(CONFIG[0])

        #Start ploting
        if self.config == CONFIG[1]:
            if len(blobs) != len(self.imgs):
                raise Exception("Blobs list must have the same length as images collection")
            self.blobs = blobs
        elif self.config == CONFIG[2]:
            if len(properties) != len(self.imgs):
                raise Exception("Properties list must have the same length as images collection")
            self.properties = properties
        elif self.config == CONFIG[3]:
            self.features = features
            self.prepare_features()
        elif self.config not in CONFIG:
            raise ValueError("Wrong value for config attribute. Please use 'standard', 'blob' or 'region' ")


        fig, ax = plt.subplots()
        self.ax_config(self.imgs, ax)

        #Set event catchers
        fig.canvas.mpl_connect('scroll_event', self.process_mouse)
        fig.canvas.mpl_connect('key_press_event', self.process_key)

    def show(self):
        # Method to call to display figure
        plt.show()

    def ax_config(self, volume, ax):
        """ Set initial configuration of the inputed axe. If config is blob, patches are added.

        :param volume: ndarray<int>
            Image to display
        :param ax: matplotlib.Axes
            Axe on which to display the image
        """
        # Check if volume shape size is 3 (i.e collection of images)
        if len(volume.shape) == 2:
            ax.volume = np.array(volume)
        elif len(volume.shape) == 3:
            ax.volume = volume
        else :
            raise Exception("Some error occurred while configuring axe for ploting. Check dimension of the image to plot")

        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        self.update_image_info(ax)

    def update_image_info(self, ax):
        if ax.patches is not None:
            [p.remove() for p in reversed(ax.patches)]

        if self.config == CONFIG[1]:
            ax.set_title("channel : {} (z = {}) : {} blobs detected".format(self.channel, ax.index, len(self.blobs[ax.index])))
            self.add_blobs_patches(ax)
        elif self.config == CONFIG[2]:
            ax.set_title(
                "channel : {} (z = {}) : {} regions detected".format(self.channel, ax.index, len(self.properties[ax.index])))
            self.add_rect_patches(ax, self.properties[ax.index])
        elif self.config == CONFIG[3]:
            ax.set_title("channel : {} (z = {}) : {} regions detected".format(self.channel, ax.index, len(self.features[ax.index])))
            self.add_rect_patches(ax, self.features[ax.index])
        else :
            ax.set_title(" channel : {} (z = {}) ".format(self.channel, ax.index))

    def add_blobs_patches(self, ax):
        blb = self.blobs[ax.index]
        # Compute radii instead of std
        blb[:, 2] = blb[:, 2] * sqrt(2)
        for blob in blb:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='blue', linewidth=2, fill=False)
            ax.add_patch(c)

    def add_rect_patches(self, ax, properties):
        cmap = cm.get_cmap('Set1')
        for index, row in properties.iterrows():  # draw rectangle around segmented coins
            c_ind = 0
            if 'label' in properties.columns:
                c_ind = row["label"]
            minr = row['bbox-0']
            minc = row['bbox-1']
            maxr = row['bbox-2']
            maxc = row['bbox-3']
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor=cmap(c_ind), linewidth=2)
            ax.add_patch(rect)

    def process_key(self, event):
        """ Function called when a keyboard event fires.

        :param event: Matplotlib.Event
            Event catched by the method fig.canvas.mpl_connect

        Note:
            Keys can be simply added or changed by modifying the conditions of this function.

        """
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.previous_slice(ax)
        elif event.key == 'k':
            self.next_slice(ax)
        fig.canvas.draw()

    def process_mouse(self, event):
        """ Function called when a mouse event fires.

                :param event: Matplotlib.Event
                    Event catched by the method fig.canvas.mpl_connect

                Note:
                    Further mouse responses such as click can be simply added or changed by modifying the conditions of this function.

                """
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.button == 'up':
            self.next_slice(ax)
        elif event.button == 'down':
            self.previous_slice(ax)
        fig.canvas.draw()

    def previous_slice(self, ax):
        """ function called to display the previous image by updating the figure.

        :param ax: Matplotlib.Axes
            Axe where the event asking for previous image fired.

        """
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
        self.update_image_info(ax)

    def next_slice(self, ax):
        """ function called to display the next image by updating the figure.

                :param ax: Matplotlib.Axes
                    Axe where the event asking for previous image fired.

                """
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        self.update_image_info(ax)

    def remove_keymap_conflicts(self, new_keys_set):
        """ function used to avoid conflicts that occured when using keybords events.
            This code is inspired from : https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data
        """
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def prepare_features(self):
        """ Update self.features by adding bounding box coordinates and spliting the DataFrame into a list of Dataframe where each line is a Z coordinate of the image.

        """
        temp = pd.DataFrame()
        for i, row in self.features.iterrows():
            depth = row['depth']
            c_x = row['centroid_x']
            c_y = row['centroid_y']
            c_z = row['centroid_z']
            area = row['area']

            #Regions are reprensented with square
            half_square_side = int(sqrt(area/depth) / 2)
            row['bbox-0'] = int(c_y - half_square_side)
            row['bbox-1'] = int(c_x - half_square_side)
            row['bbox-2'] = int(c_y + half_square_side)
            row['bbox-3'] = int(c_x + half_square_side)

            print(row)

            if depth > 1:
                # Avoid incertitude when depth is pair by increasing the depth.
                depth = depth + 1 if depth % 2 == 0 else depth
                # compute the amount of line to generate before and after z
                rad = (depth - 1)/2
                for z in range(int(c_z - rad),int(c_z +rad), 1):
                    r = row.copy()
                    r['centroid_z'] = z
                    r['depth'] = 1
                    temp = temp.append(r, ignore_index=True)
            else:
                temp = temp.append(row, ignore_index=True)

        #Split features in one dataframe per layer layers
        features = []
        for z in range(len(self.imgs)):
            features.append(temp.loc[temp['centroid_z'] == z])
        self.features = features



def display_file(file_path):
    """ Simple function to display file with the MultiLayerViewer.

    :param file_path: str
        Path of the file to display. In order to avoid any issue please enter the full adress of the file.

    """
    viewer = MultiLayerViewer(file_path)
    viewer.plot_imgs()
    #viewer.show()


def plot_single_img(image, cmap='gnuplot2', title="Undefined title"):
    """Quick image plot for debugging.

    :param image: array-like or PIL image
    :param cmap: colormap used by matplotlib
    :param title: title to display
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    fig.suptitle(title)

    if isinstance(image[0, 0], bool):
        # for binarys ploting
        ax.imshow(image, cmap='Greys', interpolation='nearest')
    else:
        ax.imshow(image, cmap=cmap)

if __name__ == '__main__':
    """FILE_NAME = "C10DsRedlessxYw_emb11_Center_Out.tif"

    PATH_TO_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
    PATH_TO_ROOT_DIR = os.path.normpath(os.path.join(PATH_TO_CURRENT_DIR, '..'))
    sys.path.append(PATH_TO_ROOT_DIR)

    DATA_PATH = os.path.join(PATH_TO_ROOT_DIR, 'data')
    EMB_PATH = os.path.join(DATA_PATH, 'embryos')
    file_path = os.path.join(EMB_PATH, FILE_NAME)"""

    emb = fm.get_tiff_file(8)
    ch1 = emb[:, :, :, 0]
    lbl_img = label_filter(ch1[0])[0]
    props = [region_properties(label_image=label_filter(img, filter=100)[0],image=img, properties=['extent', 'max_intensity', 'area', "mean_intensity", "bbox"]) for img in ch1 ]
    viewer = MultiLayerViewer(emb, channel=0)
    viewer.plot_imgs(properties=props)
    plt.show()
#    display_file(file_path)

    """
     # Code to inspire 'analysis' mode
    def ax_analysis_config(volume, axs, title = "Multislices viewers"):
        #TODO to finish (conflict to solve)
        axs[0].volume = volume
        axs.index = volume.shape[0] // 2

        img = volume[axs.index]
        label_image = label_filter(img, filter=0.10)[0]
        regions, props = region_properties(label_image, img, min_area=4,
                                           properties=['extent', 'max_intensity', 'area', "mean_intensity", "bbox"])

        def draw_rectangles(properties, picked_region=None):
            axs[0].clear()
            axs[0].imshow(volume[ax.index], cmap='gnuplot2')
            print(picked_region)
            if picked_region is not None:
                picked_minr = picked_region['bbox-0'].values
                picked_minc = picked_region['bbox-1'].values
                picked_maxr = picked_region['bbox-2'].values
                picked_maxc = picked_region['bbox-3'].values
                picked_bbox = [picked_minr, picked_minc, picked_maxr, picked_maxc]
                print(picked_bbox)
            for index, row in properties.iterrows():  # draw rectangle around segmented coins
                minr = properties['bbox-0'].iloc[index]
                minc = properties['bbox-1'].iloc[index]
                maxr = properties['bbox-2'].iloc[index]
                maxc = properties['bbox-3'].iloc[index]
                bbox = [minr, minc, maxr, maxc]

                if picked_region is not None and picked_bbox == bbox:
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='red', linewidth=2)
                else:
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False, edgecolor='blue', linewidth=2)
                axs[0].add_patch(rect)

        draw_rectangles(props)

        print(props.shape)
        print(props.head())

        points = axs[1].scatter(x=props['area'], y=props["mean_intensity"], facecolors=["C0"] * len(props),
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
            print("Fire")
            ind = event.ind
            if len(ind) > 1:
                ind = [ind[0]]
            change_point_color(ind)
            region_props_picked = props.iloc[ind]
            draw_rectangles(props, region_props_picked)

        fig.canvas.mpl_connect('pick_event', onpick)

        plt.tight_layout()
"""