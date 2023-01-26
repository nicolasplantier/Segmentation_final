# Importations 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage import filters
from skimage.filters import rank
from skimage.util import img_as_ubyte
import time
import pandas as pd 
import os
import laspy 
import pickle as pkl
import codecs


plot_width = 1500
plot_height = 1500
m = plot_width


if __name__ == "__main__": 
    df_final = pd.read_csv('df_final.csv')
    tetrapods_mask = {}
    for filename in os.listdir("./tetrapods_mask/"):
        tetrapods_mask[filename] = np.loadtxt(f"./tetrapods_mask/{filename}")
        n = tetrapods_mask[filename].shape[0]

    # We now convert the point in the dataframe to points in the image 
    xmin = df_final['x'].min()
    xmax = df_final['x'].max()
    ymin = df_final['y'].min()
    ymax = df_final['y'].max()
    df_final['i'] = (m*((df_final['x'] - xmin)/(xmax - xmin))).astype(np.int64)
    df_final['j'] = (m*((df_final['y'] - ymin)/(ymax - ymin))).astype(np.int64)
    df_final['pixel'] = df_final['i']*m + df_final['j']

    # We create the dataframe representing the image 
    pixels_list = []
    for i in range(1,m+1):
        pixels_list += (np.ones(m)*i*m + np.arange(1, m+1)).tolist()
    df_image = pd.DataFrame(np.array(pixels_list), columns=['pixel']).astype(np.int64)


    # We change the classification value 
    i = 0
    final_column_classif = np.zeros(len(df_image))
    for key, value in tetrapods_mask.items():
        df_image[f"classification_tetra_{i}"] = 0
        image_flatten = value.flatten('F').astype(np.int64)
        if len(image_flatten) == len(df_image):
            df_image.loc[:, f"classification_tetra_{i}"] = image_flatten
        i += 1
    n_tetrapods = i

    for i in range(n_tetrapods):
        df_final2 = df_final.copy()
        filename = "./3D_patch_initial.las"
        las = laspy.read(filename)
        df_final2 = df_final2.merge(df_image[df_image.loc[:, f"classification_tetra_{i}"] == 1], how='inner', on='pixel')
        
        # We need to change a bit the coordinates
        df_final2.loc[:,'x'] += np.array(las.x).min()
        df_final2.loc[:,'y'] += np.array(las.y).min()
        rap_x = las.X[0]/las.x[0]
        rap_y = las.y[0]/las.y[0]
        rap_z = las.z[0]/las.z[0]

        filter_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        header = las.header
        header.point_count = len(df_final2)
        filter_las = laspy.LasData(header)
        filter_las.points.X = df_final2['x']*rap_x
        filter_las.points.Y = df_final2['y']*rap_y
        filter_las.points.Z = df_final2['z']*rap_z
        filter_las.points.x = df_final2['x']
        filter_las.points.y = df_final2['y']
        filter_las.points.z = df_final2['z']
        filter_las.write(f"./tetrapods_models/tetrapod_model_{i}.las")
        