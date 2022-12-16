from Utils import get_gray_matter_section_polygons, open_masks, readAndStandardize
from imageio import imwrite
import json
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

model_output_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_7_-20"
mask_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION"
stats_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_7_-20/Stats"

json_files = {}
# Read all .roi and .zip 
masks = open_masks(mask_folder)
idx = 0
for file in os.listdir(model_output_folder):
    if file.endswith("_cell_count.json"):
        print(str(idx) + " out of " + str(len(masks)))
        idx += 1
        with open(os.path.join(model_output_folder, file)) as f:
            data = json.load(f)
            file = file.replace("_cell_count.json", "")
            cell_image_name = file + ".png"
            image = readAndStandardize(model_output_folder + "/" + cell_image_name)
            # plt.imshow(image)
            
            json_files[file] = data

            overlay = image.copy()

            for contour_dict in json_files[file]["countor_dicts"]:
                center = contour_dict["center"]
                print(center)
                diameter = contour_dict["equivalent_diameter"]
                cv2.circle(overlay, (int(center[0]),int(center[1])), int(diameter/2),(255,255,255), 3)
                

            alpha = 0.4  # Transparency factor.
  
            # Following line overlays transparent rectangle
            # over the image
            image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            section_counts = {}
            for key, value in masks[file].items():
                polygon = list(zip(value["x"],value["y"]))
                gray_matter_section_dict = get_gray_matter_section_polygons(polygon)
                # Check which section the center of a cell is in.
                if(gray_matter_section_dict!=None):
                    cv2.drawContours(image_new,[np.array(list(zip(gray_matter_section_dict["left_ventral_horn"].exterior.xy[0], gray_matter_section_dict["left_ventral_horn"].exterior.xy[1]))).astype(int)],0,(255,0,0),5)
                    cv2.drawContours(image_new,[np.array(list(zip(gray_matter_section_dict["left_lateral_horn"].exterior.xy[0], gray_matter_section_dict["left_lateral_horn"].exterior.xy[1]))).astype(int)],0,(255,102,0),5)
                    cv2.drawContours(image_new,[np.array(list(zip(gray_matter_section_dict["left_dorsal_horn"].exterior.xy[0], gray_matter_section_dict["left_dorsal_horn"].exterior.xy[1]))).astype(int)],0,(255,255,0),5)
                    cv2.drawContours(image_new,[np.array(list(zip(gray_matter_section_dict["right_dorsal_horn"].exterior.xy[0], gray_matter_section_dict["right_dorsal_horn"].exterior.xy[1]))).astype(int)],0,(0,255,0),5)
                    cv2.drawContours(image_new,[np.array(list(zip(gray_matter_section_dict["right_lateral_horn"].exterior.xy[0], gray_matter_section_dict["right_lateral_horn"].exterior.xy[1]))).astype(int)],0,(0,0,255),5)
                    cv2.drawContours(image_new,[np.array(list(zip(gray_matter_section_dict["right_ventral_horn"].exterior.xy[0], gray_matter_section_dict["right_ventral_horn"].exterior.xy[1]))).astype(int)],0,(255,0,255),5)
                    
            imwrite(model_output_folder + "/" + file + "_sectioned.png", image_new)