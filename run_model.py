from Utils import open_images_and_masks, mask_to_countour, normalizeImages, contour_to_mask, standardizeImage, maskImage, get_mask_number, mask_to_dots_image, dots_image_to_density
from matplotlib import pyplot as plt
from cellpose import models, core
from cellpose.io import logger_setup
from cellpose import plot
import numpy as np
import cv2
from skimage import measure
from cellpose import io
from skimage.io import imsave
import os
import json
from imageio import imwrite
from time import monotonic

file_dir = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION"
save_dir = file_dir + "/model_output"

model_path = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Coding/CellPoseTesting/split/models/CP_20221116_160410"
batch_size = 4

resume = True

use_GPU = core.use_gpu()
logger_setup()

model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)
params = [
    # [12, 1.0, 0],
    # [12, 1.0, -0.5],
    # [12, 1.0, -1.0],
    # [12, 1.0, -1.5],
    # [12, 1.0, -2.0],
    # [12, 1.0, -2.5],
    # [12, 1.0, -3.0],
    # [12, 1.0, -3.5],
    # [12, 1.0, -4.0],
    # [12, 0.9, -3.0],
    [12, 0.8, -2.0],
    # [12, 0.7, -3.0],
]
for param in params:
    density_images = []
    save_dir = file_dir + "/model_output_" + str(param[0]) + "_" + str(int(param[1]*10)) + "_" + str(int(param[2]*10))
    os.makedirs(save_dir, exist_ok=True)
    image_mask_pairs = open_images_and_masks(file_dir)
    
    if(resume):
        file_names = os.listdir(save_dir)
        images_in_target_dir, masks_in_target_dir, file_names_in_target_dir = zip(*image_mask_pairs)
        file_names_in_target_dir = list(file_names_in_target_dir)
        masks_in_target_dir = list(masks_in_target_dir)
        images_in_target_dir = list(images_in_target_dir)
        for file_name in file_names:
            if(".png" in file_name):
                file_name = file_name.replace(".png", "")
                json_name = file_name + "_cell_count.json"
                # Read count from json file.
                json_file = open(save_dir + "/" + json_name, "r")
                json_data = json.load(json_file)
                json_file.close()
                if(json_data["count"] != 0):
                    try:
                        print("output image: "  + file_name)
                        # print(file_names_in_target_dir)
                        index = file_names_in_target_dir.index(file_name)
                        print("target image: " + file_names_in_target_dir[index])
                        file_names_in_target_dir.pop(index)
                        masks_in_target_dir.pop(index)
                        images_in_target_dir.pop(index)
                        
                    except ValueError:
                        print("Value not in target directory but present in output directory")
        image_mask_pairs = list(zip(images_in_target_dir, masks_in_target_dir, file_names_in_target_dir))
            
    num_of_images = len(image_mask_pairs)
    idx = 0
    for image, mask, file_name in image_mask_pairs:
        # print(mask)
        start = monotonic()
        print("Processing image " + str(idx) + " of " + str(num_of_images) + " : " + file_name)
        idx += 1
        masks, flows, styles = model.eval([image], diameter=param[0], flow_threshold=param[1], channels=[0,0], cellprob_threshold=param[2])
        seg_mask = masks[0]

        polygons = []
        for key, value in mask.items():
            polygon = [list(zip(value["x"],value["y"]))]
            polygons.append(np.array(polygon,dtype=np.int32))
        mask_image = contour_to_mask(image.shape[0:2], polygons)
        mask_image = standardizeImage(mask_image)
        
        masked_seg_image = standardizeImage(seg_mask)
        masked_seg_image = maskImage(mask_image, masked_seg_image)
        num_of_mask = get_mask_number(masked_seg_image)

        flow_image = standardizeImage(flows[0][0])
        flow_image = maskImage(mask_image, flow_image)
        # print(flow_image.dtype)

        prob_image = standardizeImage(flows[0][1][0]) #[:,:,0]
        prob_image = maskImage(mask_image, prob_image)
        # print(prob_image.dtype)

        dots = mask_to_dots_image(masked_seg_image)
        dots_image = standardizeImage(dots)
        # print(dots_image.dtype)
        density_image = dots_image_to_density(dots, 35)
        density_images.append((file_name, density_image))
        density_image = standardizeImage(density_image, force_uint8=True)
        
        density_image_color = cv2.cvtColor(cv2.applyColorMap(density_image, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)

        density_overlay_image = cv2.addWeighted(image, 0.85, density_image_color, 0.15, 0)

        imwrite(save_dir + "/" + file_name + ".png", image)
        imwrite(save_dir + "/" + file_name + "_cell_mask.png", masked_seg_image)
        imwrite(save_dir + "/" + file_name + "_cell_flow.png", flow_image)
        imwrite(save_dir + "/" + file_name + "_cell_prob.png", prob_image)
        imwrite(save_dir + "/" + file_name + "_cell_dots.png", dots_image)
        imwrite(save_dir + "/" + file_name + "_cell_density.png", density_image)
        imwrite(save_dir + "/" + file_name + "_cell_density_overlay.png", density_overlay_image)

        with open(save_dir + "/" + file_name + "_cell_count.json", "w") as outfile:
            json.dump({"cell_count": num_of_mask}, outfile)

        end = monotonic()
        print("Processing took: " + str(end-start))

    minMaxedImages = normalizeImages([image for file_name, image in density_images], method="minmax")
    for (file_name, _), image in list(zip(density_images, minMaxedImages)):
        image = standardizeImage(image, force_uint8=False, max=1)
        imwrite(save_dir + "/" + file_name + "_cell_density_minmaxed.png", image)