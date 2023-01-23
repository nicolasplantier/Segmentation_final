# Importations 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from statsmodels.formula.api import ols 
from tqdm import tqdm 
import time
import math 


# Definning global variables
radius = 0.06
px_size = 0.10
px_size = 0.08
radius_l2 = 0.05
radius2 = 1


# Definition of the planity_measure 
def planity_measure_df(df):
    """This function takes a point as input and gives the absolute error : distance to the plane at this point"""
    if len(df) < 3:
        planity_measure = 0
    else :    
        regress_plane = ols("z ~ x + y", df).fit()
        d, a, b = regress_plane._results.params
        c = -1
        normal = np.array([a,b,c])
        normal = normal/np.linalg.norm(normal)
        cos_theta = np.abs(normal[2])
        z_predict = regress_plane.predict(df)
        dist_table = cos_theta*(np.abs(z_predict - df['z']))
        dist_table_max = dist_table.copy()
        ids = []
        for k in range(len(df)//10): 
            id_max = dist_table_max.idxmax(axis = 0)
            ids.append(id_max)
            dist_table_max.drop(index = id_max)
        dist_table.loc[ids] = 100*dist_table[ids]
        planity_measure = dist_table.std()
    return planity_measure
    

if __name__ == "__main__":

    # Start the clock 
    start = time.time()

    csv_filenames = ["..\\ajaccio_patches\\An2019-1-1-3_patch15.csv",
                    "..\\ajaccio_patches\\An2019-1-1-1_patch21.csv",
                    "..\\ajaccio_patches\\e_An2020-1-0_patch16.csv"]
    for csv_filename in tqdm(csv_filenames):
        df_coords_3d = pd.read_csv(csv_filename)

        # We create a new method, where we calculate the planity measure on voxels 
        xmax, xmin = np.max(df_coords_3d['x']), np.min(df_coords_3d['x'])
        ymax, ymin = np.max(df_coords_3d['y']), np.min(df_coords_3d['y'])
        zmax, zmin = np.max(df_coords_3d['z']), np.min(df_coords_3d['z'])
        N_x = math.floor((xmax - xmin)/px_size)
        N_y = math.floor((ymax - ymin)/px_size)
        N_z = math.floor((zmax - zmin)/px_size)
        indexs = df_coords_3d.index
        df_intermediate = df_coords_3d.copy()

        # Let's know where each point belongs (voxel number)
        df_intermediate['i'] = (N_x*((df_coords_3d['x'] - xmin)/(xmax - xmin))).astype(np.int64)
        df_intermediate['j'] = (N_y*((df_coords_3d['y'] - ymin)/(ymax - ymin))).astype(np.int64)
        df_intermediate['k'] = (N_z*((df_coords_3d['z'] - zmin)/(zmax - zmin))).astype(np.int64)
        df_intermediate['index'] = df_intermediate.index
        df_intermediate['voxel'] = (df_intermediate['i'] + df_intermediate['j']*(10**(np.log10(N_x)//1 + 1)) + df_intermediate['k']*(10**(np.log10(N_x)//1 + np.log10(N_y)//1 + 2))).astype(np.int64)
        print(f" The number of voxel that contain at least one point is {df_intermediate['voxel'].unique().size}")
        
        # We calculate the planity measure in each voxel 
        data = df_intermediate.groupby(['voxel'], group_keys=True).apply(planity_measure_df)
        data = data.to_frame(name = "planity")
        data.index = data.index.astype(np.int64)
        df_final = pd.merge(df_intermediate, data, left_on='voxel', right_index=True)
        df_final.index = df_final['index']
        df_final.sort_index(inplace = True)

        df_final.to_csv("..\\test\\" +  csv_filename[-24:-4] + 'planity.csv')

    # What time is it ? 
    print(f"It took {np.round(time.time() - start, 2)} seconds to compute")