from skimage.io import imread, imsave
import os
from math import floor, ceil
import numpy as np
from matplotlib import pyplot as plt
import cellpose

file_dir = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Manual Counts copy/CellPoseTesting"
save_dir = file_dir + "/split_448"
crop_size = (448,448)

os.makedirs(save_dir, exist_ok=True)

for file_name in os.listdir(file_dir):
    file_path = file_dir + "/" + file_name
    if(file_name.endswith(tuple([".tif", ".tiff", ".png", ".jpg", ".jpeg"]))):
        extension = file_name.split(".")[-1]
        print("Found Image: " + file_name)
        image = imread(file_path)

        if(len(image.shape)==2):
            image = np.dstack([image])
        
        if(image.shape[0]<4):
            image = np.transpose(image, (1,2,0))

        image_row = ceil(image.shape[0]/crop_size[0])
        image_col = ceil(image.shape[1]/crop_size[1])
        row_offset = floor((image.shape[0]-crop_size[0])/(image_row-1))
        col_offset = floor((image.shape[1]-crop_size[1])/(image_col-1))
        # print(row_offset, col_offset)
        for col_num in range(image_col):
            for row_num in range(image_row):
                start_pos = (row_offset*row_num, col_offset*col_num)
                end_pos = (start_pos[0]+crop_size[0],start_pos[1]+crop_size[1])
                image_crop = image[
                    start_pos[0]:end_pos[0],
                    start_pos[1]:end_pos[1],
                    :
                    ]
                save_path = save_dir + "/" + file_name.replace("." + extension, "_"+str(row_num)+"_"+str(col_num)+".png")
                imsave(save_path,image_crop, check_contrast=False)