import matplotlib.pyplot as plt
from math import sqrt
from skimage import io
import os
import sys
import numpy as np

CONFIG = ['standard', 'blob']

class MultiSliceViewer():
    def __init__(self, tiff, config = 'standard', channel = 0):
        self.set_config(config)
        self.config = config
        self.channel = channel
        if isinstance(tiff, str):
            tiff = io.imread(file_path)
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

    def plot_imgs(self, blobs = None):
        #TODO : Cleanup plot_imgs function
        if len(self.imgs.shape) == 2:
            self.imgs = np.array(self.imgs)
        self.remove_keymap_conflicts({'j', 'k'})
        if blobs is not None :
            self.set_config(CONFIG[1])
        else :
            self.set_config(CONFIG[0])
        if self.config == 'standard':
            fig, ax = plt.subplots()
            self.ax_config(self.imgs, ax)
        elif self.config == 'blob':
            if blobs is None :
                raise ValueError("No blobs data while blob config is setted")
            if len(blobs) != len(self.imgs):
                raise Exception("Blobs list hasn't the same length as image")
            self.blobs = blobs
            fig, ax = plt.subplots()
            self.ax_config(self.imgs, ax)

        elif self.config == 'analysis' :
            fig, axs = plt.subplots(ncols=2, figsize=(10, 6))
            # ax_analysis_config(self.imgs, axs)

        else :
            raise ValueError("Wrong value for config attribute. Please use 'standard' or 'analysis' ")
        fig.canvas.mpl_connect('scroll_event', self.process_mouse)
        fig.canvas.mpl_connect('key_press_event', self.process_key)
    """
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

    def show(self):
        plt.show()

    def ax_config(self, volume, ax):
        if len(volume.shape) == 2:
            ax.volume= np.array(volume)
        elif len(volume.shape) == 3:
            ax.volume = volume
        else :
            raise Exception("Some error occured while configuring axe for ploting. Check dimension of the image to plot")
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        if self.config == CONFIG[1]:
            ax.set_title("channel : {} (z = {}) : {} blobs detected".format(self.channel, ax.index, len(self.blobs[ax.index])))
            self.add_blobs_patches(ax)
            ax.set_axis_off()
        else :
            ax.set_title(" channel : {} (z = {}) ".format(self.channel, ax.index))

    def add_blobs_patches(self, ax):
        blb = self.blobs[ax.index]
        #Compute radii instead of std
        blb[:, 2] = blb[:, 2] * sqrt(2)
        for blob in blb:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='blue', linewidth=2, fill=False)
            ax.add_patch(c)

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.previous_slice(ax)
        elif event.key == 'k':
            self.next_slice(ax)
        fig.canvas.draw()

    def process_mouse(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.button == 'up':
            self.next_slice(ax)
        elif event.button == 'down':
            self.previous_slice(ax)
        fig.canvas.draw()

    def previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
        if self.config != CONFIG[0]:
            [p.remove() for p in reversed(ax.patches)]
            if self.config == CONFIG[1]:
                ax.set_title("channel : {} (z = {}) : {} blobs detected".format(self.channel, ax.index,len(self.blobs[ax.index])))
                self.add_blobs_patches(ax)
        else :
            ax.set_title(" channel : {} (z = {}) ".format(self.channel, ax.index))

    def next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        if self.config != CONFIG[0]:
            [p.remove() for p in reversed(ax.patches)]
            if self.config == CONFIG[1]:
                ax.set_title("channel : {} (z = {}) : {} blobs detected".format(self.channel, ax.index, len(self.blobs[ax.index])))
                self.add_blobs_patches(ax)
        else :
            ax.set_title(" channel : {} (z = {}) ".format(self.channel, ax.index))

    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)



def display_file(file_path):
    #TODO : check if it works
    viewer = MultiSliceViewer(file_path)
    viewer.plot_imgs()


def plot_single_img(image, cmap='gnuplot2', title="Undefined title"):
    """
    Quick image plot for debugging.
    :param image: array-like or PIL image
    :param cmap: colormap used by matplotlib
    :param title: title to display
    :return:
    """
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    fig.suptitle(title)

    if isinstance(image[0, 0], bool):
        # for binarys ploting
        ax.imshow(image, cmap='Greys', interpolation='nearest')
    else:
        ax.imshow(image, cmap=cmap)

if __name__ == '__main__':
    FILE_NAME = "C10DsRedlessxYw_emb11_Center_Out.tif"

    PATH_TO_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
    PATH_TO_ROOT_DIR = os.path.normpath(os.path.join(PATH_TO_CURRENT_DIR, '..'))
    sys.path.append(PATH_TO_ROOT_DIR)

    DATA_PATH = os.path.join(PATH_TO_ROOT_DIR, 'data')
    file_path = os.path.join(DATA_PATH, FILE_NAME)

    display_file(file_path)

