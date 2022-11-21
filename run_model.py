from Utils import open_images_and_masks, mask_to_countour, contour_to_mask, standardizeImage, maskImage, get_mask_number, mask_to_dots_image, dots_image_to_density
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

file_dir = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Coding/TO TEST CODE"
save_dir = file_dir + "/model_output"

model_path = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Coding/CellPoseTesting/split/models/CP_20221116_160410"
batch_size = 4

use_GPU = core.use_gpu()
logger_setup()
os.makedirs(save_dir, exist_ok=True)

model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)
params = [
    [12, 1.0, 0],
    [12, 1.0, -0.5],
    [12, 1.0, -1.0],
    [12, 1.0, -1.5],
    [12, 1.0, -2.0],
    [12, 1.0, -2.5],
    [12, 1.0, -3.0],
    [12, 1.0, -3.5],
    [12, 1.0, -4.0],
    [12, 0.9, -3.0],
    [12, 0.8, -3.0],
    [12, 0.7, -3.0],
]
for param in params:
    save_dir = file_dir + "/model_output_" + param[0] + "_" + param[1] + "_" + param[2]
    image_mask_pairs = open_images_and_masks(file_dir)
    for image, mask, file_name in image_mask_pairs:
        masks, flows, styles = model.eval([image], diameter=param[0], flow_threshold=param[1], channels=[0,0], cellprob_threshold=param[2])
        seg_mask = masks[0]

        polygons = []
        for key, value in mask.items():
            polygon = [list(zip(value["x"],value["y"]))]
            polygons.append(polygon)
        mask_image = contour_to_mask(image.shape[0:2], polygons)
        mask_image = standardizeImage(mask_image)
        
        masked_seg_image = standardizeImage(seg_mask)
        masked_seg_image = maskImage(mask_image, masked_seg_image)
        num_of_mask = get_mask_number(masked_seg_image)

        flow_image = standardizeImage(flows[0][0])
        flow_image = maskImage(mask_image, flow_image)
        print(flow_image.dtype)

        prob_image = standardizeImage(flows[0][1][0]) #[:,:,0]
        prob_image = maskImage(mask_image, prob_image)
        print(prob_image.dtype)

        dots = mask_to_dots_image(masked_seg_image)
        dots_image = standardizeImage(dots)
        print(dots_image.dtype)
        density_image = dots_image_to_density(dots, 35)
        density_image = standardizeImage(density_image, force_uint8=True)

        density_image_color = cv2.cvtColor(cv2.applyColorMap(density_image, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)

        density_overlay_image = cv2.addWeighted(image, 0.85, density_image_color, 0.15, 0)
        
        # plt.imshow(density_overlay_image)
        # plt.imshow(density_image)
        # plt.show()

        imwrite(save_dir + "/" + file_name + ".png", image)
        imwrite(save_dir + "/" + file_name + "_cell_mask.png", masked_seg_image)
        imwrite(save_dir + "/" + file_name + "_cell_flow.png", flow_image)
        imwrite(save_dir + "/" + file_name + "_cell_prob.png", prob_image)
        imwrite(save_dir + "/" + file_name + "_cell_dots.png", dots_image)
        imwrite(save_dir + "/" + file_name + "_cell_density.png", density_image)
        imwrite(save_dir + "/" + file_name + "_cell_density_overlay.png", density_overlay_image)

        with open(save_dir + "/" + file_name + "_cell_count.json", "w") as outfile:
            json.dump({"cell_count": num_of_mask}, outfile)

    # images = []
    # for idx, cropped_image in enumerate(generateCrops(file_dir, save_dir, crop_size, False)):
    #     images.append(cropped_image)
    #     if((idx+1)%batch_size==0):
    #         masks, flows, styles = model.eval(images, diameter=None, flow_threshold=1.0, channels=[0,0], cellprob_threshold=0.3)
    #         for iidx in range(batch_size):
    #             maski = masks[iidx]
    #             flowi = flows[iidx][0]

    #             fig = plt.figure(figsize=(12,5))
                # plot.show_segmentation(fig, images[iidx], maski, flowi, channels=[0,0])
    #             plt.tight_layout()
    #             plt.show()
    #         images = []