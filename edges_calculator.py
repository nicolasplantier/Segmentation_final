# Importations 

import numpy as np 
import matplotlib.pyplot as plt 
import cv2 as cv 
import laspy
from laspy.file import File
from scipy.spatial import cKDTree
import pandas as pd
import seaborn as sns
from PIL import Image 
import colorcet as cc
from statsmodels.formula.api import ols 
from tqdm import tqdm 
import time
import datashader as ds 
import seaborn as sns
from datashader.utils import export_image
import scipy
import math


path = "./Documents/Mines Paris/2A/Data sophia/Projet de recherche/Tetrapodes/Segmentation/"
execute = "python create_image_edges.py"
delta = 0.011
kernel_size_max = 4

plot_width = 1500
plot_height = 1500

if __name__ == "__main__": 

    # Read the .las file and import the data 
    filename = "./3d_patch_classified_test.las"
    las = laspy.read(filename) 
    n = len(las.x)
    x_scaled = np.array(las.x)
    y_scaled = np.array(las.y) 
    z_scaled = np.array(las.z)
    np_coords_3d = np.concatenate((x_scaled.reshape((n,1)), y_scaled.reshape((n,1)), z_scaled.reshape((n,1)), np.array(las.points.planity[:].tolist()).reshape((n,1))), axis = 1)
    df_coords_3d = pd.DataFrame(data=np_coords_3d, columns=['x', 'y', 'z', 'planity']) 

    # Let's delete the points behind, that we can't see
    N_ini_points = len(df_coords_3d)
    xmax = np.max(df_coords_3d['x'])
    xmin = np.min(df_coords_3d['x'])
    ymax = np.max(df_coords_3d['y'])
    ymin = np.min(df_coords_3d['y'])
    N_x = math.floor((xmax - xmin)/delta)
    N_y = math.floor((ymax - ymin)/delta)
    create_image = np.zeros((N_x, N_y))
    indexs = df_coords_3d.index

    df_intermediate = df_coords_3d.copy()
    df_intermediate['i'] = (N_x*((df_coords_3d['x'] - xmin)/(xmax - xmin))).astype(np.int64)
    df_intermediate['j'] = (N_y*((df_coords_3d['y'] - ymin)/(ymax - ymin))).astype(np.int64)
    df_coords_3d['pixel'] = df_intermediate['i']*N_y + df_intermediate['j'] # On utilise l'unicité de la division euclidienne de k par Nc
    idx = df_coords_3d.groupby(['pixel'])['z'].transform(max) == df_coords_3d['z']
    df_coords_3d = df_coords_3d[idx] # The new dataframe is ready and much smaller 
    print(f"We kept {np.round(100*len(df_coords_3d)/N_ini_points,2)} % of the initial points : the rest of them are not visible")

    # We now want a good scale for planity measure (we want the maximum value to be taken by more than a few points)
    fig, axes = plt.subplots(1, 2) 
    axes[0].hist(np.array(df_coords_3d['planity']).flatten(), bins = 100)
    axes[0].set_title("Planity measure historgram before rescale")
    max_value = 0.70 #We fix it at 0.7 (empirical measure)
    mask = (df_coords_3d['planity'] > max_value)
    index = df_coords_3d[mask].index
    df_coords_3d.loc[index, 'planity'] = max_value
    axes[1].hist(np.array(df_coords_3d['planity']).flatten(), bins = 100)
    axes[1].set_title("Planity measure historgram before rescale")
    plt.savefig("planity_measure_historgram.png")
    plt.close()


    """ # Create camera parameters
    x_med = (df_coords_3d['x'].max() + df_coords_3d['x'].min())/2
    y_med = (df_coords_3d['y'].max() + df_coords_3d['y'].min())/2
    z_med = (df_coords_3d['z'].max() + df_coords_3d['z'].min())/2

    tvec = np.array([x_med, y_med, z_med + 800], dtype = np.float64)
    rvec = np.array([0,0,0], dtype = np.float64)
    cmatrix =  np.identity(3, dtype = np.float64)

    # On fait maintenant la projection de tous nos points
    image_proj = cv.projectPoints(objectPoints = coords, rvec = rvec, tvec = tvec, cameraMatrix = cmatrix, distCoeffs= None)[0]

    # On crée la dataframe de l'image 
    X = image_proj[:, 0, 0]
    Y = image_proj[:, 0, 1]
    image_proj_df = pd.DataFrame(X, columns = ['x'])
    image_proj_df['y'] = Y
    image_proj_df['x'] = (image_proj_df['x'] - min(X))/(max(X) - min(X))
    image_proj_df['y'] = (image_proj_df['y'] - min(Y))/(max(Y) - min(Y))

    delta_x = max(X) - min(X)
    delta_y = max(Y) - min(Y)
    N_x = np.int32(60000*(max(X) - min(X)))
    N_y = np.int32(60000*(max(Y) - min(Y)))"""

    # We now create the final image
    cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height)
    agg = cvs.points(df_coords_3d, 'x', 'y', agg = ds.reductions.mean('planity'))
    agg_array = np.asarray(agg)
    np.nan_to_num(agg_array, copy=False)
    max_array = agg_array.max()

    # We first set maximum value to 0.45
    # agg_array[agg_array > 0.45] = 0.45

    # We then apply a maximum function to avoid black pixels : it reduces the size of the shape but makes better the classification algorithm
    agg_array = scipy.ndimage.maximum_filter(input  = agg_array, size = kernel_size_max, mode = 'constant') 
    
    # We save the txt representing the image 
    np.savetxt("table_image2.txt", agg_array)

    plt.imshow(agg_array, cmap = 'jet')
    plt.savefig('image_egdes_scaled_caché', dpi = 800)
    print("The finale fig is saved")
