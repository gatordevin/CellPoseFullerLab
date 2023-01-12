import os
import json
import xlsxwriter
from Utils import open_masks, readAndStandardize, mask_to_json, get_gray_matter_section_polygons
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd

model_output_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_7_-20"

json_files = {}
# Read all .roi and .zip
idx = 0
for file in os.listdir(model_output_folder):
    if file.endswith("_cell_masks_info.json"):
        file_name = file.replace("_cell_masks_info.json", "")

        cell_mask_info = {}
        with open(os.path.join(model_output_folder, file)) as f:
            data = json.load(f)
            json_files[file_name] = data
            cell_mask_info = data

        image_name = file.replace("_cell_masks_info.json", ".png")

        image = readAndStandardize(model_output_folder + "/" + image_name)

        contours = []
        for contour_dict in cell_mask_info["countor_dicts"]:
            contour = contour_dict["contour"]
            contour = np.array(contour)
            contour = contour.reshape((-1, 1, 2))
            contours.append(contour)

        print("Number of contours: " + str(len(contours)))

        outlined_cells_image = cv2.drawContours(image, contours, -1, (255, 255, 255), 1)

        # save image.
        cv2.imwrite(model_output_folder + "/" + file_name + "_outlined_cells.png", outlined_cells_image)

        filled_cells_image = cv2.drawContours(image, contours, -1, (255, 255, 255), -1)

        # save image.
        cv2.imwrite(model_output_folder + "/" + file_name + "_filled_cells.png", filled_cells_image)