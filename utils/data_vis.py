''' 
Data visualization 
'''

import matplotlib.pyplot as plt
from matplotlib import patches
from utils.data_processing import csv_to_df
from utils.global_const import COL_NAMES, SAVE_FIG, DPI

#######################################################################################
# Data visualization

def get_cmap(num, name='tab20c'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, num)

def plot_distribution(data_frame, col_name, 
                      info, path, filt=None):
    ''' Plot a barchart for a column in a dataframe '''
    x_label, y_label, title = info
    plt.rcdefaults()
    val_counts = data_frame[col_name].value_counts()
    # print(val_counts)
    if filt:
        val_counts = val_counts[val_counts >= filt]
    idx_list = val_counts.index.tolist()
    val_list = val_counts.values.tolist()
    cmap = get_cmap(len(idx_list))
    color_list = [cmap(idx) for idx in range(len(idx_list))]

    fig, axs = plt.subplots(figsize=(10, 6))
    chart = axs.barh(idx_list, val_list, color=color_list)
    axs.set_title(title)
    axs.set_xlabel(x_label)
    axs.set_ylabel(y_label)
    axs.invert_yaxis()
    axs.bar_label(chart)
    if SAVE_FIG:
        fig.savefig(path + title + '.pdf')
    return fig

def plot_boxes(jpg_name, bbx_name, title, path):
    ''' Plot image with annotated boxes '''
    image = plt.imread(jpg_name)
    num_row, num_col, dummy_channel = image.shape
    annos = csv_to_df(bbx_name, COL_NAMES).to_dict()

    fig, axs = plt.subplots(figsize=(num_col / DPI, num_row / DPI), dpi=DPI)  
    axs.imshow(image, origin='lower')
    axs.set_axis_off()
    axs.set_title(title)
    # draw rectangles
    for idx in range(len(annos["x"])):
        rect = patches.Rectangle((annos["x"][idx], annos["y"][idx]), 
            annos["width"][idx], annos["height"][idx], 
            linewidth=0.5, edgecolor='b', facecolor='none')
        axs.add_patch(rect)

    if SAVE_FIG:
        fig.savefig(path + title + '.jpg')
    return fig
