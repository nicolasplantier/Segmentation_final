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
            image_flatten[image_flatten == 1] += i # We put a different classification for each tetrapod
            df_image.loc[:, f"classification_tetra_{i}"] = image_flatten
            final_column_classif += image_flatten
        i += 1
    df_image['classification'] = final_column_classif.astype(np.int64)
    df_final = df_final.merge(df_image, how='inner', on='pixel')

    # We now change the cloud points
    filename = "./3D_patch_initial.las"
    las = laspy.read(filename)
    
    # We need to change a bit the coordinates
    df_final.loc[:,'x'] += np.array(las.x).min()
    df_final.loc[:,'y'] += np.array(las.y).min()
    rap_x = las.X[0]/las.x[0]
    rap_y = las.y[0]/las.y[0]
    rap_z = las.z[0]/las.z[0]

    # las.add_extra_dim(laspy.ExtraBytesParams(name="tetrapod_classification",type=np.float64,description="More classes available"))


    filter_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    header = las.header
    header.point_count = len(df_final)
    filter_las = laspy.LasData(header)
    filter_las.points.X = df_final['x']*rap_x
    filter_las.points.Y = df_final['y']*rap_y
    filter_las.points.Z = df_final['z']*rap_z
    filter_las.points.x = df_final['x']
    filter_las.points.y = df_final['y']
    filter_las.points.z = df_final['z']
    filter_las.add_extra_dim(laspy.ExtraBytesParams(name="tetrapod_classification",type=np.float64,description="More classes available"))
    filter_las.points.tetrapod_classification[:] = df_final['classification']
    filter_las.write('3d_patch_classified_final.las')