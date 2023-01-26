"""
Aim of this .py file :
Create an hypercube from a .las cloudpoints : high values represent voxel where normal vectors (surface of the tetrapod) converge.
We will then deduce where the axes of the tetrapod are. 
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
m = 100 #we want a final cube of size 1000x1000x1000 (which is a lot)


def create_3d_image(m : int):
    """This function creates a 3d image : a column voxel is added as well as i,j,k columns"""
    pixels_list = []
    i_table = np.arange(1, m+1)
    j_table = (np.arange(1, m+1)*(10**(np.log10(m)//1 + 1))).astype(np.int64)
    k_table = (np.arange(1, m+1)*(10**(np.log10(m)//1 + np.log10(m)//1 + 2))).astype(np.int64)
    X, Y, Z = np.meshgrid(i_table, j_table, k_table)
    df_image_3d = pd.DataFrame(np.array(X+Y+Z).flatten(), columns=['voxel']).astype(np.int64)

    cent = 10**(np.log10(m)//1 + np.log10(m)//1 + 2)
    diz = 10**(np.log10(m)//1 + 1)
    unit = 1
    df_image_3d['i'] = ((df_image_3d['voxel'] - cent*(df_image_3d['voxel']//cent) - diz*((df_image_3d['voxel'] - cent*(df_image_3d['voxel']//cent))//diz))//unit).astype(np.int64)
    df_image_3d['j'] = ((df_image_3d['voxel'] - cent*(df_image_3d['voxel']//cent))//diz).astype(np.int64)
    df_image_3d['k'] = (df_image_3d['voxel']//cent).astype(np.int64)
    return df_image_3d


def create_voxel_column_constant(df, m :int): 
    """
    m is the number of voxel that we want in x, y and z. So it is better if the image has the same size in x, y and z.
    Otherwise, it is better to use create_voxel_column, but it is more complicated to build the 3d image. 
    """
    xmin = df['x'].min()
    xmax = df['x'].max()
    ymin = df['y'].min()
    ymax = df['y'].max()
    zmin = df['z'].min()
    zmax = df['z'].max()
    df['i'] = (m*((df_coords_3d['x'] - xmin)/(xmax - xmin))).astype(np.int64)
    df['j'] = (m*((df_coords_3d['y'] - ymin)/(ymax - ymin))).astype(np.int64)
    df['k'] = (m*((df_coords_3d['z'] - zmin)/(zmax - zmin))).astype(np.int64)
    df['index'] = df.index
    df['voxel'] = (df['i'] + df['j']*(10**(np.log10(m)//1 + 1)) + df['k']*(10**(np.log10(m)//1 + np.log10(m)//1 + 2))).astype(np.int64)
    return df

def vect_plan(df):
    """This function takes points in the same voxel and returns the norm of the vector on the x,y plan"""
    if len(df) < 3:
        normal = np.nan
    else :    
        regress_plane = ols("z ~ x + y", df).fit()
        d, a, b = regress_plane._results.params
        c = -1
        normal = np.array([a,b,c])
        normal = delta*(normal/np.linalg.norm(normal))
    return np.round(normal,5)

def calculate_delta(df):
    # Let's calculate the norm that we will need for the gradient 
    xmin = df['x'].min()
    xmax = df['x'].max()
    ymin = df['y'].min()
    ymax = df['y'].max()
    zmin = df['z'].min()
    zmax = df['z'].max()
    delta = np.round(min((xmax-xmin)/m, (ymax-ymin)/m, (zmax-zmin)/m), 2) # This is the approximate size of a voxel (delta x delta x delta)
    xsize = (xmax-xmin)/m
    ysize = (ymax-ymin)/m
    zsize = (zmax-zmin)/m
    return delta, xsize, ysize, zsize


def create_voxel_list(df):
    """
    This function will calculate all the voxels that we be touched by the vector planity.
    The i,j,k coordinates are the coordinates of the point where we calculated the vector_planity. 
    """
    # We first add the i,j,k columns
    df['voxel'] = df.index
    cent = 10**(np.log10(m)//1 + np.log10(m)//1 + 2)
    diz = 10**(np.log10(m)//1 + 1)
    unit = 1
    df['i'] = ((df['voxel'] - cent*(df['voxel']//cent) - diz*((df['voxel'] - cent*(df['voxel']//cent))//diz))//unit).astype(np.int64)
    df['j'] = ((df['voxel'] - cent*(df['voxel']//cent))//diz).astype(np.int64)
    df['k'] = (df['voxel']//cent).astype(np.int64)
    df.drop('voxel', axis = 1, inplace = True)

    # We first need to convert the vector_planity in (x,y,z) to (i,j,k)
    df['dn_x'] = df.loc[:, 'dx']/xsize
    df['dn_y'] = df.loc[:, 'dy']/ysize
    df['dn_z'] = df.loc[:, 'dz']/zsize
    df.drop('dx', axis = 1, inplace = True)
    df.drop('dy', axis = 1, inplace = True)
    df.drop('dz', axis = 1, inplace = True)

    # We will just add the first 2, that is the i,j,k method
    """n = 2
    for k in range(1,n+1):
        df[f"next_voxel_{k}_i"] = (df[f"i"] + k*df['dn_x']).astype(np.int64)
        df[f"next_voxel_{k}_j"] = (df[f"j"] + k*df['dn_y']).astype(np.int64)
        df[f"next_voxel_{k}_k"] = (df[f"k"] + k*df['dn_z']).astype(np.int64)"""

    # We will just add the first 2, that is the voxel method
    n = 30
    for k in range(1,n+1):
        mask_i = (0 <= df[f"i"] + k*df['dn_x']) & (df[f"i"] + k*df['dn_x']<= m) #we need to be careful that we are still inside the cube in the x direction
        mask_j = (0 <= df[f"j"] + k*df['dn_y']) & (df[f"j"] + k*df['dn_y']<= m) #we need to be careful that we are still inside the cube in the y direction
        mask_k = (0 <= df[f"k"] + k*df['dn_z']) & (df[f"k"] + k*df['dn_z']<= m) #we need to be careful that we are still inside the cube in the z direction
        mask = mask_i & mask_j & mask_k
        df[f"voxel_{k}"] = -1
        df.loc[:,f"voxel_{k}"][mask] = ((df[mask][f"i"] + k*df[mask]['dn_x']).astype(np.int64)*unit + (df[mask][f"j"] + k*df[mask]['dn_y']).astype(np.int64)*diz + (df[mask][f"k"] + k*df[mask]['dn_z']).astype(np.int64)*cent).astype(np.int64)
    for k in range(2,n+1):
        mask = (df[f"voxel_{k}"] == df[f"voxel_{k-1}"])
        df.loc[:,f"voxel_{k}"][mask] = -1
    arr = np.unique(np.array(df.iloc[:,6:6+n]), return_counts=True)
    voxels = arr[0]
    counts = arr[1]
    df_voxel_counts = pd.DataFrame(index=voxels, data = counts, columns=['counts'])
    return df, df_voxel_counts

# ------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__": 
    for filename in os.listdir("./tetrapods_models"):
        if filename != ".DS_Store":
            f = f"./tetrapods_models/{filename}"
            las = laspy.read(f)
            n = len(las.x)

            # shift the data to simplify
            x_scaled = np.array(las.x) - np.array(las.x).min()
            y_scaled = np.array(las.y) -  np.array(las.y).min()
            z_scaled = np.array(las.z) 


            # Create the dataframe
            np_coords_3d = np.concatenate((x_scaled.reshape((n,1)), y_scaled.reshape((n,1)), z_scaled.reshape((n,1))), axis = 1)
            df_coords_3d = pd.DataFrame(data=np_coords_3d, columns=['x', 'y', 'z']) 


            # We create the new voxel column as well as i,j,k, index, voxel columns
            delta, xsize, ysize, zsize = calculate_delta(df_coords_3d)
            df_coords_3d = create_voxel_column_constant(df_coords_3d, m)
            df_image_3d = create_3d_image(m)



            # ----------------------------------------------------------------------------------------------------------------


            # We calculate the planity measure in each voxel 
            data = df_coords_3d.groupby(['voxel'], group_keys=True).apply(vect_plan)
            data = data.to_frame(name = "planity_vector") # in this dataframe, we have the normal vector for each voxel => we need to calculate 
            data.dropna(inplace = True)

            # just to split the columns 
            data['voxel'] = data.index
            data = (pd.concat([data['voxel'], data['planity_vector'].apply(pd.Series)], axis = 1).rename(columns = {0: 'dx', 1: 'dy', 2: 'dz'})).drop('voxel', axis = 1) 
            data, df_voxel_counts = create_voxel_list(data)
            df_image_3d.index = df_image_3d['voxel']
            df_final_image = df_image_3d.join(df_voxel_counts) 
            df_final_image = df_final_image.fillna(0).astype(np.int64)

            df_final_image_copy = df_final_image.copy()
            """df_final_image_copy['j_f'] = m-df_final_image['j']
            df_final_image_copy['i_f'] = df_final_image['i']

            df_final_image.loc[:,'i'] = df_final_image_copy.loc[:,'i_f']      
            df_final_image.loc[:,'j'] = df_final_image_copy.loc[:,'j_f']"""  

            fig, axes = plt.subplots(3,3,figsize=(10, 6))
            s = 0
            for i in range(3):
                for j in range(3):
                    s += 10
                    image = df_final_image[df_final_image['k'] == s] # we go in the middle of the cube
                    cvs = ds.Canvas(plot_width=m, plot_height=m)
                    agg = cvs.points(image, 'i', 'j', agg = ds.reductions.mean('counts'))
                    agg_array = np.asarray(agg)
                    np.nan_to_num(agg_array, copy=False)
                    axes[i,j].set_title(f"Cut over the layer number {s}")
                    axes[i,j].axis('off')
                    axes[i,j].imshow(agg_array, cmap = 'jet')

            plt.savefig(f"./echographie_tetrapods/{filename[:-4]}.png", dpi = 400)
            df_final_image.to_csv(f"./echographie_tetrapods/{filename[:-4]}.csv")



            