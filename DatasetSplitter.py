from skimage.io import imread, imsave
import os
from math import floor, ceil
import numpy as np
from matplotlib import pyplot as plt

def tile_image(img, shape, overlap_percentage):
    x_points = [0]
    stride = int(shape[0] * (1-overlap_percentage))
    counter = 1
    while True:
        pt = stride * counter
        if pt + shape[0] >= img.shape[0]:
            if shape[0] == img.shape[0]:
                break
            x_points.append(img.shape[0] - shape[0])
            break
        else:
            x_points.append(pt)
        counter += 1
    
    y_points = [0]
    stride = int(shape[1] * (1-overlap_percentage))
    counter = 1
    while True:
        pt = stride * counter
        if pt + shape[1] >= img.shape[1]:
            if shape[1] == img.shape[1]:
                break
            y_points.append(img.shape[1] - shape[1])
            break
        else:
            y_points.append(pt)
        counter += 1
    splits = []
    for i in y_points:
        for j in x_points:
            split = img[i:i+shape[0], j:j+shape[1]]
            splits.append([split, (i, j)])
    return splits

def readAndStandardize(file_path):
    image = imread(file_path)
    if(len(image.shape)==2):
        image = np.dstack([image])
    
    if(image.shape[0]<4):
        image = np.transpose(image, (1,2,0))
    return image

def generateCrops(file_dir, save_dir, crop_size, save_images=True):
    os.makedirs(save_dir, exist_ok=True)
    for file_name in os.listdir(file_dir):
        file_path = file_dir + "/" + file_name
        if(file_name.endswith(tuple([".tif", ".tiff", ".png", ".jpg", ".jpeg"]))):
            extension = file_name.split(".")[-1]
            print("Found Image: " + file_name)
            image = readAndStandardize(file_path)
            
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
                    if(save_images):
                        save_path = save_dir + "/" + file_name.replace("." + extension, "_"+str(row_num)+"_"+str(col_num)+".png")
                        imsave(save_path,image_crop, check_contrast=False)
                    yield image_crop