''' 
Data processing.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#######################################################################################
# Data processing

def get_file_names(path, extension):
    ''' 
    Get a list of file names with certain extension in the folder dir.
    '''
    file_names = []
    for file in os.listdir(path):
        if file.endswith(extension):
            file_names.append(os.path.join(path, file))
    return sorted(file_names)

def csv_to_df(file_name, cols):
    ''' 
    Read a CSV file and save it to a dataframe.
    Input:
        file_name - a string that is the file name
    Returns a dataframe containing rows in the CSV file, and the first row is the header.
    '''
    data_df = pd.read_csv(file_name, header=0, names=cols)
    return data_df

def concat_frames(file_names, cols):
    ''' Concat dataframes saved in a list of file names into a single dataframe. ''' 
    frames = [csv_to_df(file_name, cols) for file_name in file_names]
    return pd.concat(frames, axis=0, ignore_index=True)

def read_jpg(file_name):
    ''' 
    Read a JPG file and save it to a np array.
    Input:
        file_name - a string that is the file name
    Returns a np array containing RGB values of the image.
    '''
    image = plt.imread(file_name)
    # print(image.shape)
    return image

def add_col(frame, added_col_name, col_name, values_dict):
    "Add a column to a dataframe based on a mapping dictionary. "
    frame[added_col_name] = frame[col_name].map(values_dict)
    return frame

def split_img_annos(img_files, anno_files, frac, seed=None):
    ''' 
    Split the data into three parts 
    Copy the files into output path
    '''
    if seed:
        np.random.seed(seed)

    num_of_files = len(img_files)
    train_idx = int(num_of_files * frac[0])
    test_idx = train_idx + int(num_of_files * frac[1])

    indices = np.arange(num_of_files)
    np.random.shuffle(indices)
    train_test_val = [
        {'jpg': [img_files[idx] for idx in indices[:train_idx]],
         'csv': [anno_files[idx] for idx in indices[:train_idx]]},
        {'jpg': [img_files[idx] for idx in indices[train_idx: test_idx]],
         'csv': [anno_files[idx] for idx in indices[train_idx: test_idx]]},
        {'jpg': [img_files[idx] for idx in indices[test_idx:]],
         'csv': [anno_files[idx] for idx in indices[test_idx:]]},
    ]

    return train_test_val

def coordinate_to_box(x_1, y_1, width, height):
    ''' Compute box coordinate ''' 
    x_2, y_2 = x_1 + width, y_1 + height
    box = [x_1, y_1, x_2, y_2]

    return box

def cropping():
    ''' Crop image bird only and resize into fixed shape '''
    pass
