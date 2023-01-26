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
import os 


# Global variables 
m = 100
#m = 150


if __name__ == "__main__": 
    for filename in os.listdir("./echographie_tetrapods"):
        if filename != ".DS_Store" and filename[-4:] == '.csv':
            f = f"./echographie_tetrapods/{filename}"
            df_final_image = pd.read_csv(f)
            """df_final_image.loc[:,'i'] = m - df_final_image.loc[:,'i']
            df_final_image.loc[:,'j'] = m - df_final_image.loc[:,'j']"""
            #df_final_image = pd.read_csv(f"echographie_all_tetrapods.csv")
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
            np.savetxt(f"./tetrapods_points_axes_to_find/point_cloud_{filename[:-4]}.dat", delimiter = ",", X = point_cloud, fmt="%d") # fmt = "%d" to have int 

        # NB : to execute hough3dlines : "./hough3dlines test -o out" in the hough-3d-lines folder 