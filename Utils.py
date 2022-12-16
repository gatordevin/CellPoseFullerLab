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
from shapely.geometry import Point, LineString, Polygon
from shapely import affinity
from skimage import measure
from skimage import io
from scipy import ndimage
from scipy import stats
from time import monotonic
from skimage.segmentation import find_boundaries
from shapely.ops import linemerge, polygonize, unary_union

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
    

def standardizeImage(image, force_uint8=False, max=None):
    if(max==None):
        max = image.max()
    if(image.dtype != np.uint8):
        if(len(np.unique(image)[1:])>256):
            image = image.astype(np.float32)
            image += image.min()
            if(force_uint8):
                image/=(max / 256)
                image = image.astype(np.uint8)
            else:
                image/=(max / 65535)
                image = image.astype(np.uint16)
        else:
            image = image.astype(np.uint8)
    if(len(image.shape)==2):
        image = np.dstack([image])
    
    if(image.shape[0]<4):
        image = np.transpose(image, (1,2,0))

    if(image.shape[-1]==2):
        empty_img = np.zeros(image[:,:,0].shape)
        image = np.dstack([image, empty_img])
    return image

def get_gray_matter_section_polygons(polygon):
    shapely_polygon = Polygon(polygon)
    convex_hull = shapely_polygon.convex_hull

    lines = []
    for i in range(len(convex_hull.exterior.xy[0])-1):
        lines.append(LineString([Point(convex_hull.exterior.xy[0][i], convex_hull.exterior.xy[1][i]), Point(convex_hull.exterior.xy[0][i+1], convex_hull.exterior.xy[1][i+1])]))
    lines.sort(key=lambda x: x.length, reverse=True)
    if(len(lines) > 3):
        longest_line = lines[0]
    else:
        return

    # Get all lines longer than 100 pixels.
    lines = [line for line in lines if line.length > 150]
    # Find the two lines that hav the most similiar angle.
    smallest_angle_lines = []
    angle_diff = []
    for i in range(len(lines)):
        angle1 = np.arctan2(lines[i].coords[1][1] - lines[i].coords[0][1], lines[i].coords[1][0] - lines[i].coords[0][0])
        for j in range(i+1, len(lines)):
            # Check if the two lines share a point.
            if(lines[i].coords[0] == lines[j].coords[0] or lines[i].coords[0] == lines[j].coords[1] or lines[i].coords[1] == lines[j].coords[0] or lines[i].coords[1] == lines[j].coords[1]):
                continue
            angle2 = np.arctan2(lines[j].coords[1][1] - lines[j].coords[0][1], lines[j].coords[1][0] - lines[j].coords[0][0])
            # print(angle1, angle2)
            dff = abs(angle1 - angle2)
            # Minimize the difference between the two angles.
            if(dff > np.pi/2):
                dff = np.pi - dff
            # to degrees.
            dff = abs(np.degrees(dff))
            # print(dff)
            angle_diff.append(dff)
            # check if dff is the smallest in angle diff and if so add the two lines to the smallest_angle_lines.
            if(dff == min(angle_diff)):
                smallest_angle_lines = [lines[i], lines[j]]
            if(len(smallest_angle_lines) == 0):
                smallest_angle_lines = [lines[i], lines[j]]

    if(len(smallest_angle_lines) == 0):
        # Display shapely_polygon.
        print("Invalid polygon moving on to next polygon.")
        return

    # Get average angle of smallest angle lines.
    # print(smallest_angle_lines)
    angle1 = np.arctan2(smallest_angle_lines[0].coords[1][1] - smallest_angle_lines[0].coords[0][1], smallest_angle_lines[0].coords[1][0] - smallest_angle_lines[0].coords[0][0])
    angle2 = np.arctan2(smallest_angle_lines[1].coords[1][1] - smallest_angle_lines[1].coords[0][1], smallest_angle_lines[1].coords[1][0] - smallest_angle_lines[1].coords[0][0])
    angle = (angle1 + angle2)/2
    angle = np.degrees(angle)
    print("Average angle of smallest angle lines:", angle)

    # Find distance two smallest lines are apart.
    distance = smallest_angle_lines[0].distance(smallest_angle_lines[1])
    print("Distance between smallest angle lines:", distance)

    # Get line between midpoints of smallest angle lines.
    midpoint1 = smallest_angle_lines[0].interpolate(0.5, normalized=True)
    midpoint2 = smallest_angle_lines[1].interpolate(0.5, normalized=True)
    midpoint_line = LineString([midpoint1, midpoint2])
    # Plot midpoint1 and midpoint2.
    

    # Rotate shapely polygon around center of midpoint line use the average angle of the smallest angle lines.
    center = midpoint_line.interpolate(0.5, normalized=True)
    shapely_polygon = affinity.rotate(shapely_polygon, -angle, origin=center)
    midpoint_line = affinity.rotate(midpoint_line, -angle, origin=center)
    midpoint1 = affinity.rotate(midpoint1, -angle, origin=center)
    midpoint2 = affinity.rotate(midpoint2, -angle, origin=center)
    plt.plot(shapely_polygon.exterior.xy[0], shapely_polygon.exterior.xy[1], color='r')
    plt.plot(midpoint_line.coords.xy[0], midpoint_line.coords.xy[1], color='g')
    # plt.plot(midpoint1.x, midpoint1.y, 'ro')
    # plt.plot(midpoint2.x, midpoint2.y, 'bo')

    # Plot two lines one at the same angle and one perpendicular to the angle.
    line1 = affinity.rotate(midpoint_line, 90, "centroid")
    # Increase size of line1

    # Plot two other lines that are the same anlge but shifted by distance/6.
    # Translate line 1 along line 2 a distance of distance/6.
    xoff = distance/6 * np.cos(np.radians(angle))
    yoff = distance/6 * np.sin(np.radians(angle))
    # Line 3 should be translated by xoff and yoff in the direction of midpoint 1.
    # Found the direction of midpoint 1 as 1 or -1 from the center of the midpoint line.
    direction = 1
    if(midpoint1.x < center.x):
        direction = -1
    line3 = affinity.translate(line1, xoff*direction, yoff*direction)
    line4 = affinity.translate(line1, xoff*-direction, yoff*-direction)

    # Plot the two lines that are perpendicular to the angle.
    plt.plot(line3.coords.xy[0], line3.coords.xy[1], color='b')
    # plt.plot(line4.coords.xy[0], line4.coords.xy[1], color='b')

    line3 = affinity.scale(line3, 1000, 1000)
    line4 = affinity.scale(line4, 1000, 1000)

    merged = linemerge([shapely_polygon.boundary, midpoint_line])
    borders = unary_union(merged)
    polygons = polygonize(borders)

    left_gray_matter = polygons[0]
    right_gray_matter = polygons[1]

    try:

        boundary = left_gray_matter.boundary
        if(left_gray_matter.boundary.geom_type=="MultiLineString"):
            longest = None
            for linestring in left_gray_matter.boundary.geoms:
                if(longest==None):
                    longest = linestring
                else:
                    if(linestring.length>longest.length):
                        longest = linestring
            boundary = longest
        

        merged = linemerge([boundary, line3])
        borders = unary_union(merged)
        left_polygons = polygonize(borders)

        boundary = right_gray_matter.boundary
        if(right_gray_matter.boundary.geom_type=="MultiLineString"):
            longest = None
            for linestring in right_gray_matter.boundary.geoms:
                if(longest==None):
                    longest = linestring
                else:
                    if(linestring.length>longest.length):
                        longest = linestring
            boundary = longest

        merged = linemerge([boundary, line3])
        borders = unary_union(merged)
        right_polygons = polygonize(borders)

        plt.plot(left_polygons[0].exterior.xy[0], left_polygons[0].exterior.xy[1], color='g')
        left_dorsal_horn = left_polygons[1]
        right_dorsal_horn = right_polygons[1]

        plt.cla()

        boundary = left_polygons[0].boundary
        if(left_polygons[0].boundary.geom_type=="MultiLineString"):
            longest = None
            for linestring in left_polygons[0].boundary.geoms:
                if(longest==None):
                    longest = linestring
                else:
                    if(linestring.length>longest.length):
                        longest = linestring
            boundary = longest

        merged = linemerge([boundary, line4])
        borders = unary_union(merged)
        left_polygons = polygonize(borders)

        boundary = right_polygons[0].boundary
        if(right_polygons[0].boundary.geom_type=="MultiLineString"):
            longest = None
            for linestring in right_polygons[0].boundary.geoms:
                if(longest==None):
                    longest = linestring
                else:
                    if(linestring.length>longest.length):
                        longest = linestring
            boundary = longest

        merged = linemerge([boundary, line4])
        borders = unary_union(merged)
        right_polygons = polygonize(borders)

        left_ventral_horn = left_polygons[0]
        right_ventral_horn = right_polygons[0]

        left_lateral_horn = left_polygons[1]
        right_lateral_horn = right_polygons[1]

        # Rotate saved polygons back.
        left_gray_matter = affinity.rotate(left_gray_matter, angle, origin=center)
        right_gray_matter = affinity.rotate(right_gray_matter, angle, origin=center)
        left_dorsal_horn = affinity.rotate(left_dorsal_horn, angle, origin=center)
        right_dorsal_horn = affinity.rotate(right_dorsal_horn, angle, origin=center)
        left_ventral_horn = affinity.rotate(left_ventral_horn, angle, origin=center)
        right_ventral_horn = affinity.rotate(right_ventral_horn, angle, origin=center)
        left_lateral_horn = affinity.rotate(left_lateral_horn, angle, origin=center)
        right_lateral_horn = affinity.rotate(right_lateral_horn, angle, origin=center)

        polygon_dict = {
            "left_dorsal_horn" : left_dorsal_horn,
            "right_dorsal_horn" : right_dorsal_horn,
            "left_ventral_horn" : left_ventral_horn,
            "right_ventral_horn" : right_ventral_horn,
            "left_lateral_horn" : left_lateral_horn,
            "right_lateral_horn" : right_lateral_horn
        }

        return polygon_dict
    except:
        return None

def normalizeImages(images, method="minmax"):
    means = []
    maxs = []
    mins = []
    stds = []
    for image in images:
        means.append(np.mean(image))
        maxs.append(np.max(image))
        mins.append(np.min(image))
    mean = np.mean(means)
    max = np.max(maxs)
    min = np.min(mins)

    for image in images:
        stds.append(np.std(image))

    std = np.mean(stds)
    
    print(mean, std)
    output_images = []
    for image in images:
        if(method=="minmax"):
            output_images.append(image / max)
        elif(method=="normal"):
            output_images.append((image-mean)/std)

    return output_images

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
                # print("reading roi")
                roi = read_roi_file(file_dir + "/" + mask_name)
                new_values = None
                for key, value in roi.items():
                    value : dict
                    if "x" not in list(value.keys()):
                        # print(value)
                        # print("processing roi")   
                        new_values = {}
                        # print(len(value["paths"]))
                        for idx, path in enumerate(value["paths"]):
                            new_values[str(idx)] = {}
                            new_values[str(idx)]["x"] = []
                            new_values[str(idx)]["y"] = []
                            for coord in path:
                                new_values[str(idx)]["x"].append(coord[0])
                                new_values[str(idx)]["y"].append(coord[1])
                if(new_values!=None):
                    roi = new_values
                # print(roi)
            else:
                # print("reading zip")
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

def open_masks(file_dir):
    image_mask_set = {}
    file_names = os.listdir(file_dir)
    for file_name in file_names:
        roi = None
        if(file_name.endswith("_Mask.roi")):
            # print("Opening: " + file_name)
            roi = read_roi_file(file_dir + "/" + file_name)
            for key, value in roi.items():
                value : dict
                if "x" not in list(value.keys()):
                    # print(value)
                    # print("processing roi")
                    new_values = {}
                    new_values["x"] = []
                    new_values["y"] = []
                    for coord in value["paths"][0]:
                        new_values["x"].append(coord[0])
                        new_values["y"].append(coord[1])
                    roi[key] = new_values
            file_name = file_name.replace("_Mask.roi", "")
            image_mask_set[file_name] = roi
        if(file_name.endswith("_Mask.zip")):
            # print("Opening: " + file_name)
            try:
                roi = dict(read_roi_zip(file_dir + "/" + file_name))
            except BadZipFile:
                roi = None
                print("ROI file is not a zip file please remask: " + file_dir + "/" + file_name)
            file_name = file_name.replace("_Mask.zip", "")
            image_mask_set[file_name] = roi
    return image_mask_set

def mask_to_countour(mask):
    print(mask.shape)

def dots_image_to_density(dots_image, kernel_size):
    dots_image = dots_image.astype(np.float32)
    img = ndimage.gaussian_filter(dots_image[:,:,0], (kernel_size,kernel_size))
    return img

def mask_to_dots_image(mask):
    blank = np.zeros(mask.shape)

    mask[find_boundaries(mask, mode='inner')] = 0
    contours, hierarchy = cv2.findContours((mask==0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[1:]

    center_points = []
    # Compute statistics and append to dictionary.
    for c in contours:
        if(len(c) < 5):
            continue
        mass = cv2.moments(c)
        center = [mass["m10"] / mass["m00"], mass["m01"] / mass["m00"]]
        center_points.append(center)
        blank[int(center[0]),int(center[1]), 0] += 1

    print("Converting mask to dots")
    return blank

def mask_to_json(mask):
    mask[find_boundaries(mask, mode='inner')] = 0
    contours, hierarchy = cv2.findContours((mask==0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = contours[1:]

    contour_dicts = []
    # Compute statistics and append to dictionary.
    for c in contours:
        if(len(c) < 5):
            continue
        contour_dict = {}
        # Center of mass.
        mass = cv2.moments(c)
        contour_dict["center"] = [mass["m10"] / mass["m00"], mass["m01"] / mass["m00"]]
        contour_dict["area"] = cv2.contourArea(c)
        contour_dict["perimeter"] = cv2.arcLength(c, True)
        contour_dict["contour"] = c.tolist()
        contour_dict["circularity"] = 4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, True) ** 2
        try:
            contour_dict["eccentricity"] = cv2.fitEllipse(c)[1][0] / cv2.fitEllipse(c)[1][1]
        except:
            print(c)
            plt.imshow(mask)
            plt.plot(c[:,0,0], c[:,0,1])
            plt.show()
        contour_dict["orientation"] = cv2.fitEllipse(c)[2]
        contour_dict["major_axis_length"] = cv2.fitEllipse(c)[1][0]
        contour_dict["minor_axis_length"] = cv2.fitEllipse(c)[1][1]
        contour_dict["solidity"] = cv2.contourArea(c) / cv2.contourArea(cv2.convexHull(c))
        contour_dict["extent"] = cv2.contourArea(c) / (cv2.fitEllipse(c)[1][0] * cv2.fitEllipse(c)[1][1])
        contour_dict["equivalent_diameter"] = np.sqrt(4 * cv2.contourArea(c) / np.pi)
        contour_dicts.append(contour_dict)

    return contour_dicts

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