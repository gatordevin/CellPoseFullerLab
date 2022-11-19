from zipfile import BadZipFile
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
import os
from math import floor, ceil
import numpy as np
from matplotlib import pyplot as plt
from read_roi import read_roi_file, read_roi_zip
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage import io

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

def standardizeImage(image):
    if(len(image.shape)==2):
        image = np.dstack([image])
    
    if(image.shape[0]<4):
        image = np.transpose(image, (1,2,0))

    if(image.shape[-1]==2):
        empty_img = np.zeros(image[:,:,0].shape)
        image = np.dstack([image, empty_img])
    return image

def maskImage(grayscale_mask, image):
    mask = grayscale_mask
    if(len(image.shape)>2):
        image_stack = []
        for i in range(image.shape[2]):
            image_stack.append(grayscale_mask)
        mask = np.dstack(image_stack)
    masked_image = image * mask
    return masked_image

def readAndStandardize(file_path):
    image = imread(file_path)
    image = standardizeImage(image)
    
    return image

def grayscale_to_seg_array(grayscale_mask):
    contours = []
    for pixel_val in np.unique(grayscale_mask)[1:]:
        single_seg_mask = np.where(grayscale_mask == pixel_val, 1, 0)
        c = measure.find_contours(single_seg_mask)
        contours.append(c)

    contours = [contour[0] for contour in contours]
    return contours

def get_mask_number(grayscale_mask):
    # print(np.unique(grayscale_mask))
    num_of_mask = len(np.unique(grayscale_mask)[1:])
    return num_of_mask

def open_images_and_masks(file_dir, image_ext=[".tiff",".tif"]):
    image_mask_set = []
    file_names = os.listdir(file_dir)
    for file_name in file_names:
        if(file_name.endswith(tuple(image_ext))):
            file_ext = "." + file_name.split(".")[-1]
            image_path = file_dir + "/" + file_name
            roi = None
            mask_name = file_name.replace(file_ext, "_Mask.roi")
            if(mask_name in file_names):
                roi = read_roi_file(file_dir + "/" + mask_name)
            else:
                mask_name = file_name.replace(file_ext, "_Mask.zip")
                if(mask_name in file_names):
                    try:
                        roi = dict(read_roi_zip(file_dir + "/" + mask_name))
                    except BadZipFile:
                        roi = None
                        print("ROI file is not a zip file please remask: " + file_dir + "/" + mask_name)
            image = readAndStandardize(image_path)
            image_mask_set.append((image, roi, file_name.replace(file_ext, "")))
    return image_mask_set

def mask_to_countour(mask):
    print(mask.shape)

def contour_to_mask(mask_shape, polygons):
    mask_image = np.zeros(mask_shape, dtype=np.uint8)
    polygons = [np.array(polygon) for polygon in polygons]
    value = 1
    cv2.fillPoly(mask_image, polygons, value)
    return mask_image

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