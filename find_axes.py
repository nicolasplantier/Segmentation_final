"""
Aim of this .py file :
Find the axes into the 3d cube.
"""

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
import math 
import random as rd 
from planity_calculator import create_voxel_column
import datashader as ds 
from mpl_toolkits.mplot3d import axes3d  # Fonction pour la 3D
from matplotlib import style


# Global variables 
m = 100


if __name__ == "__main__": 
    df_final_image = pd.read_csv(f"echographie_tetrapod.csv")
    df_final_image.drop("voxel.1", axis = 1, inplace = True)

    # Initialisation
    mask = (df_final_image['k'] == 100)
    image = df_final_image[mask]
    cvs = ds.Canvas(plot_width=m, plot_height=m)
    agg = cvs.points(image, 'i', 'j', agg = ds.reductions.mean('counts'))
    agg_array = np.asarray(agg)
    np.nan_to_num(agg_array, copy=False)
    agg_array = agg_array.astype(np.int64)
    i_max, j_max = np.where(agg_array == np.max(agg_array))

    p_max_value = np.max(agg_array)
    p_i_max, p_j_max = i_max[0], j_max[0] #p for previous 

    X = []
    Y = []
    Y_der = [0]
    axe_points_coordinates = []

    """for k in tqdm(range(99, 0, -1)):  # We go layers after layers 
        if Y_der[-1] <= 7 or k >= 90:
            X.append(k)
            mask = (df_final_image['k'] == k)
            image = df_final_image[mask]
            cvs = ds.Canvas(plot_width=m, plot_height=m)
            agg = cvs.points(image, 'i', 'j', agg = ds.reductions.mean('counts'))
            agg_array = np.asarray(agg)
            np.nan_to_num(agg_array, copy=False)
            agg_array = agg_array.astype(np.int64)

            max_value = np.max(agg_array)
            i_max, j_max = np.where(agg_array == max_value)
            i_max, j_max = i_max[0], j_max[0]

            Y.append(np.linalg.norm(np.array([i_max,j_max]) - np.array([p_i_max, p_j_max])))
            p_i_max = i_max
            p_j_max = j_max
            if k != 99: 
                Y_der.append(Y[-1]-Y[-2]) # this is the derivative of the function
            if Y_der[-1] <= 7:
                axe_points_coordinates.append([p_i_max, p_j_max, k])
        else : 
            break"""

    for k in tqdm(range(99, 0, -1)):  # We go layers after layers 
        mask = (df_final_image['k'] == k)
        image = df_final_image[mask]
        cvs = ds.Canvas(plot_width=m, plot_height=m)
        agg = cvs.points(image, 'i', 'j', agg = ds.reductions.mean('counts'))
        agg_array = np.asarray(agg)
        np.nan_to_num(agg_array, copy=False)
        agg_array = agg_array.astype(np.int64)
        max_value = np.max(agg_array)
        limit_max_value = 0.7*max_value
        if limit_max_value >= 12: # we need to find out for import points that are in an axe 
            i,j = np.where(agg_array >= limit_max_value)
            axe_points_coordinates += [[i[l],j[l], k] for l in range(len(i))]

    point_cloud = np.array(axe_points_coordinates).astype(np.int64)
    np.savetxt("point_cloud_main_axe2.dat", delimiter = ",", X = point_cloud, fmt="%d") # fmt = "%d" to have int 

# NB : to execute hough3dlines : "./hough3dlines test -o out" in the hough-3d-lines folder 