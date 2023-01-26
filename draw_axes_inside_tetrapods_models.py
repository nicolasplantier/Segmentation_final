"""
This aim of this file is to find out where the are inside the tetrapods model and integrate them inside the .las file. 
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

m = 100  #this is the size of the cube where we found the axes for instance 

if __name__ == "__main__":
    # we first need to import the tetrapod model that we want
    filename = f"./tetrapods_models/tetrapod_model_2.las"
    las = laspy.read(filename) 
    n = len(las.x)

    # shift the data to simplify
    x_scaled = np.array(las.x) - np.array(las.x).min()
    y_scaled = np.array(las.y) -  np.array(las.y).min()
    z_scaled = np.array(las.z) 

    # Create the dataframe
    np_coords_3d = np.concatenate((x_scaled.reshape((n,1)), y_scaled.reshape((n,1)), z_scaled.reshape((n,1))), axis = 1)
    df_coords_3d = pd.DataFrame(data=np_coords_3d, columns=['x', 'y', 'z']) 


    df_axes_plot =pd.DataFrame(columns = ['i','j','k']) 


    # adding the line plot
    # creatin plot_list :
    plot_list = []
    
    text = open(f"./axes_tetrapods/main_axes_point_cloud_tetrapod_model_2.dat").readlines()
    n_lines = len(text)

    text_line = text[0]
    n = eval(text_line[text_line.find("npoints=")+8:text_line.find(", ")])
    a = eval(text_line[text_line.find("a=")+2:text_line.find("),")+1])
    b = eval(text_line[text_line.find("b=")+2:]) 
    plot_list.append([np.array(list(a)), np.array(list(b))])
    
    for k in range(1,n_lines):
        text_line = text[k]
        new_n = eval(text_line[text_line.find("npoints=")+8:text_line.find(", ")])
        new_a = eval(text_line[text_line.find("a=")+2:text_line.find("),")+1])
        new_b = eval(text_line[text_line.find("b=")+2:]) 

        # print(np.linalg.norm(np.array(plot_list)[:,1]-np.array(list(new_b)), axis = 1))
        intermediate_array = np.linalg.norm((np.array(plot_list)[:,1]-np.array(list(new_b))), axis = 1)
        mask_direction = all((intermediate_array > 1))

        mask_n_points = (new_n > 15)
        if mask_direction and mask_n_points:
                plot_list.append([np.array(list(new_a)), np.array(list(new_b))])

    for element in plot_list:
        a = element[0]
        b = element[1]/100
        vect_list = []
        for k in range (25*100): 
                vect_list.append(a+k*b)
        for k in range (25*100): 
                vect_list.append(a-k*b)
        vect_list = np.array(vect_list)

        df_vect_list =  pd.DataFrame(data = vect_list, columns = ['i', 'j', 'k'])
        df_axes_plot = pd.concat([df_axes_plot, df_vect_list], axis = 0)


    # We now need to convert this coordinates in i,j,k to x,y,z
    xmin = df_coords_3d['x'].min()
    xmax = df_coords_3d['x'].max()
    ymin = df_coords_3d['y'].min()
    ymax = df_coords_3d['y'].max()
    zmin = df_coords_3d['z'].min()
    zmax = df_coords_3d['z'].max()

    df_axes_plot.loc[:,'if'] = df_axes_plot['i']
    df_axes_plot.loc[:,'jf'] = df_axes_plot['j']


    df_axes_plot['x']  = xmin + (xmax - xmin)*(df_axes_plot['if']/m)
    df_axes_plot['y']  = ymin + (ymax - ymin)*(df_axes_plot['jf']/m)
    df_axes_plot['z']  = zmin + (zmax - zmin)*(df_axes_plot['k']/m)


    max_index = df_coords_3d.index.max()
    df_axes_plot.index = np.arange(max_index+1, max_index+1 + len(df_axes_plot))
    df_points_to_add = pd.DataFrame(data = df_axes_plot[['x','y','z']], columns = ['x', 'y', 'z'])
    df_coords_3d = pd.concat([df_coords_3d,df_points_to_add], axis = 0)


    
    # We need to change a bit the coordinates
    df_coords_3d.loc[:,'x'] += np.array(las.x).min()
    df_coords_3d.loc[:,'y'] += np.array(las.y).min()
    rap_x = las.X[0]/las.x[0]
    rap_y = las.y[0]/las.y[0]
    rap_z = las.z[0]/las.z[0]

    filter_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    header = las.header
    header.point_count = len(df_coords_3d)
    filter_las = laspy.LasData(header)
    filter_las.points.X = df_coords_3d['x']*rap_x
    filter_las.points.Y = df_coords_3d['y']*rap_y
    filter_las.points.Z = df_coords_3d['z']*rap_z
    filter_las.points.x = df_coords_3d['x']
    filter_las.points.y = df_coords_3d['y']
    filter_las.points.z = df_coords_3d['z']
    filter_las.write(f"./models_with_axes/model0.las")


