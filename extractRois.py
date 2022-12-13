import os
import json
from Utils import readAndStandardize, mask_to_json
from matplotlib import pyplot as plt
import numpy as np
import cv2

model_output_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_7_-20"
file_names = os.listdir(model_output_folder)
for idx, file_name in enumerate(file_names):
    print("Processing file: " + file_name + " (" + str(idx) + "/" + str(len(file_names)) + ")")
    if("_cell_mask.png" in file_name):
        base_image_name = file_name.replace("_cell_mask.png", "")
        cell_count_json_file_name = base_image_name + "_cell_count.json"
        cell_mask_file_name = base_image_name + "_cell_mask.png"

        # Read json file
        json_file = open(model_output_folder + "/" + cell_count_json_file_name, "r")
        image_count_info = json.load(json_file)
        json_file.close()

        # Read image
        cell_masks = readAndStandardize(model_output_folder + "/" + cell_mask_file_name)

        countor_dicts = mask_to_json(cell_masks)

        image_count_info["countor_dicts"] = countor_dicts

        # Write json file
        json_file = open(model_output_folder + "/" + cell_count_json_file_name, "w")
        json.dump(image_count_info, json_file)
        json_file.close()

