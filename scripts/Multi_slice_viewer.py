import matplotlib.pyplot as plt
from skimage import io
import numpy as np

def display_file(file_path):
    im = io.imread(file_path)
    multi_slice_viewer(im)
    plt.show()


def multi_slice_viewer(volume):
    if len(volume.shape) == 4 :
        volume = [volume]
    remove_keymap_conflicts({'j', 'k'})
    for i in np.arange(len(volume)):
        fig, ax = plt.subplots()
        ax_config(volume[i],ax)
        fig.canvas.mpl_connect('scroll_event', process_mouse)
        fig.canvas.mpl_connect('key_press_event', process_key)

def ax_config(volume,ax):
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])



def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()


def process_mouse(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'up':
        next_slice(ax)
    elif event.button == 'down':
        previous_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
