# Importations 

import numpy as np 
import laspy
from laspy.file import File
import pandas as pd
import scipy
import time
from tqdm import tqdm
import os

# global variables
selected_index = [1, 2, 3, 4, 5, 6, 7]
step = 0.5
min_size = 500000
points_per_patch = 1500000

# global functions
def las_to_df(las):    
    n = len(las.x)
    x_scaled = np.array(las.x) - np.array(las.x).min()
    y_scaled = np.array(las.y) -  np.array(las.y).min()
    z_scaled = np.array(las.z)
    np_coords_3d = np.concatenate((x_scaled.reshape((n,1)), y_scaled.reshape((n,1)), z_scaled.reshape((n,1))), axis = 1)
    return pd.DataFrame(data=np_coords_3d, columns=['x', 'y', 'z']) 

def for_patches(df):
    """
    Dataset x,y projection is in a 'losange' shape
    convert it canonic base to simplify the splitting
    in some cases doesn't work but nevermind
    """
    df = df.copy()
    x_min = df['x'].min()
    x_max = df['x'].max()
    y_min = df['y'].min()
    y_max = df['y'].max()
    #extremal points of the dataset
    west = df.loc[df['x'] == x_min].iloc[0][['x','y']]
    east = df.loc[df['x'] == x_max].iloc[0][['x','y']]
    south = df.loc[df['y'] == y_min].iloc[0][['x','y']]
    north = df.loc[df['y'] == y_max].iloc[0][['x','y']]
    #create losange base vectors adapt them to encompasse the whole point cloud
    f_1 = south - west
    f_3 = east - north
    f_2 = north - west
    f_4 = east - south
    norm_f_1 = np.linalg.norm(f_1)
    norm_f_3 = np.linalg.norm(f_3)
    if norm_f_3 > norm_f_1:
        f_1 = f_1/norm_f_1*norm_f_3
        norm_f_1 = np.linalg.norm(f_1)
    norm_f_2 = np.linalg.norm(f_2)
    norm_f_4 = np.linalg.norm(f_4)
    if norm_f_4 > norm_f_2:
        f_2 = f_2/norm_f_2*norm_f_4
        norm_f_2 = np.linalg.norm(f_2)
    # basis matrix from canonic to losange
    from_canonic = np.array([f_1,f_2]).T
    to_canonic = scipy.linalg.inv(from_canonic)
    #convert coordiantes to canonic
    canonized = to_canonic@(df[['x', 'y']] - west).T
    df[['x_los', 'y_los']] = canonized.T
    return df, norm_f_1, norm_f_2

def create_patches(for_patches, step = 0.8, min_size = 500000, points_per_patch = 1000000):
    """
    With for_patches' result, returns df patches of around 1 million points
    from a grid with overlapping (step designates where the following patch
    start from the preceding one)
    min_size is the minimum number of points per patch
    """
    df_can, norm_f_1, norm_f_2 = for_patches
    df_can = df_can.copy()
    n = len(df_can)
    #choose the adequate splitting
    density = n  #in the canonic basis
    patch_area = points_per_patch/density
    ratio = norm_f_2/norm_f_1
    side_x = np.sqrt(ratio * patch_area)
    side_y = side_x/ratio
    # nb of subdivision for each axis
    nb_x = 1//(step*side_x) + (1%step*side_x >0)
    nb_y = ratio*nb_x
    patches = []
    for i in range(int(nb_x) - 1):
        for j in range(int(nb_y) - 1):
            patch = df_can[(df_can['x_los'] >= i*step*side_x) &
                    (df_can['y_los'] >= j*step*side_y) &
                    (df_can['x_los'] < (i*step*side_x + side_x)) &
                    (df_can['y_los'] < (j*step*side_y + side_y))]
            if len(patch) > min_size:
                patches.append(patch[['x', 'y', 'z']])
    for i in range(int(nb_x) - 1):
        patch = df_can[(df_can['x_los'] >= i*step*side_x) &
                    (df_can['y_los'] >= 1 - side_y) &
                    (df_can['x_los'] < (i*step*side_x + side_x))]
        if len(patch) > min_size:
            patches.append(patch[['x', 'y', 'z']])
    for j in range(int(nb_y) - 1):
        patch = df_can[(df_can['x_los'] >= 1 - side_x) &
                    (df_can['y_los'] >= j*step*side_y) &
                    (df_can['y_los'] < (j*step*side_y + side_y))]
        if len(patch) > min_size:
            patches.append(patch[['x', 'y', 'z']])
    patch = df_can[(df_can['x_los'] >= 1 - side_x) &
                    (df_can['y_los'] >= 1 - side_y)]
    if len(patch) > min_size:
        patches.append(patch)
    return patches



if __name__ == "__main__": 
    # Start the clock 
    start = time.time()
    
    """
    # Read the .las file and import the data to df
    filenames = ["..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-0-0-1_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-0-0-2_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-0-0-3_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-0-0-4_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-1-1-1_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-1-1-2_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-1-1-3_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-1-0_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2020-1-0_tetrapods.las"]
    pointclouds = [laspy.read(filenames[i]) for i in selected_index]
    dfs = [las_to_df(las) for las in pointclouds]
    t_1 = np.round(time.time() - start, 2)
    print('loading time', t_1)
    current_directory = os.getcwd()
    new_directory = r'ajaccio_patches'
    final_directory = os.path.join(current_directory, new_directory)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for k in tqdm(range(len(dfs))):
        patches = create_patches(for_patches(dfs[k]), step = step)
        for number in tqdm(range(len(patches))):
            patches[number].to_csv(new_directory +  '\\' +filenames[selected_index[k]][-26:-13] + f'patch{number}.csv')
    """
    filenames = ["..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-0-0-1_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-0-0-2_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-0-0-3_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-0-0-4_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-1-1-1_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-1-1-2_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-1-1-3_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2019-1-0_tetrapods.las",
            "..\\ajaccio_tetrapods_samples\\NGF_L93_Photogrametrie_An2020-1-0_tetrapods.las"]
    pointclouds = [laspy.read(filenames[i]) for i in selected_index]
    current_directory = os.getcwd()
    new_directory = r'..\ajaccio_patches'
    final_directory = os.path.join(current_directory, new_directory)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    for k in tqdm(range(len(pointclouds))):
        df = las_to_df(pointclouds[k])
        patches = create_patches(for_patches(df), step = step, min_size = min_size, points_per_patch = points_per_patch)
        las = laspy.read(filenames[k])
        header = las.header
        rap_x = las.X[0]/las.x[0]
        rap_y = las.y[0]/las.y[0]
        rap_z = las.z[0]/las.z[0]
        for number in tqdm(range(len(patches))):
            df = patches[number]
            header.point_count = len(df)
            patch_las = laspy.LasData(header)
            df.loc[:,'x'] += np.array(las.x).min()
            df.loc[:,'y'] += np.array(las.y).min()
            patch_las.points.X = df['x']*rap_x
            patch_las.points.Y = df['y']*rap_y
            patch_las.points.Z = df['z']*rap_z
            patch_las.points.x = df['x']
            patch_las.points.y = df['y']
            patch_las.points.z = df['z']
            patch_las.write(new_directory +  '\\' +filenames[selected_index[k]][-26:-13] + f'patch{number}.las')
    # What time is it ? 
    print(np.round(time.time() - start, 2))