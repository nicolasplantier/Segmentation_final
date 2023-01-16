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


# Definning global variables
radius = 0.06
radius_l2 = 0.05
radius2 = 1


# Definition of the planity_measure 
def planity_measure(point, radius):
    """This function takes a point as input and gives the absolute error : distance to the plane at this point"""
    mask_x = (radius > point['x'] - df_final['x']) & (radius > df_final['x'] - point['x'])
    mask_y = (radius > point['y'] - df_final['y']) & (radius > df_final['y'] - point['y'])
    mask_z = (radius > point['z'] - df_final['z']) & (radius > df_final['z'] - point['z'])
    mask_neighboorhood = mask_x &  mask_y & mask_z
    
    #mask_neighboorhood = ((((df_final[['x', 'y', 'z']] - point)**2).sum(axis = 1)) < radius_l2)
    neighborhood = df_final[mask_neighboorhood]

    if len(neighborhood) <= 3:
        planity_measure = 1
    else :
        regress_plane = ols("z ~ x + y", neighborhood).fit()
        d, a, b = regress_plane._results.params
        c = -1
        normal = np.array([a,b,c])
        normal = normal/np.linalg.norm(normal)
        cos_theta = np.abs(normal[2])
        z_predict = regress_plane.predict(neighborhood)
        # planity_measure = np.abs((cos_theta*(np.abs(z_predict - neighborhood['z']))).mean())
        dist_table = cos_theta*(np.abs(z_predict - neighborhood['z']))
        dist_table_max = dist_table.copy()
        ids = []
        for k in range(len(neighborhood)//10): 
            id_max = dist_table_max.idxmax(axis = 0)
            ids.append(id_max)
            dist_table_max.drop(index = id_max)
        dist_table.loc[ids] = 100*dist_table[ids]
        planity_measure = dist_table.std()
        #print(planity_measure)

    index = df_final[mask_neighboorhood].index
    df_final.loc[index,'planity_classifier'] = planity_measure

    return index
    

if __name__ == "__main__":

    # Start the clock 
    start = time.time()

    # Read the .las file 
    filename = "./3D_patch_initial.las"
    las = laspy.read(filename) 
    n = len(las.x)

    # shift the data to simplify
    x_scaled = np.array(las.x) - np.array(las.x).min()
    y_scaled = np.array(las.y) -  np.array(las.y).min()
    z_scaled = np.array(las.z) 

    # Create the dataframe
    np_coords_3d = np.concatenate((x_scaled.reshape((n,1)), y_scaled.reshape((n,1)), z_scaled.reshape((n,1))), axis = 1)
    df_coords_3d = pd.DataFrame(data=np_coords_3d, columns=['x', 'y', 'z']) 


    # We create the intermediate dataframes to calculate the planity measure 
    df_to_be_classified = df_coords_3d.copy()
    df_final = df_coords_3d.copy()
    df_to_be_classified['planity_classifier'] = 0
    df_final['planity_classifier'] = 0

    initial_lengh = len(df_to_be_classified)
    while len(df_to_be_classified) > 1:
        print(np.round(100*(1 - len(df_to_be_classified)/initial_lengh),2), end = "\r")
        point = df_to_be_classified.iloc[0]
        index = planity_measure(point, radius = radius)
        df2 = df_final.loc[index]
        final_index = pd.merge(df_to_be_classified, df2, left_index=True, right_index=True).index
        df_to_be_classified.drop(final_index, axis= 0, inplace = True)

    # What time is it ? 
    print(np.round(time.time() - start), 2)

    index = df_final.index
    las.add_extra_dim(laspy.ExtraBytesParams(name="planity",type=np.float64,description="More classes available"))
    las.points.planity[:] = df_final['planity_classifier']

    las.write('3d_patch_classified_test.las')

