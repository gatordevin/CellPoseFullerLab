import os
import json
import xlsxwriter
from Utils import open_masks
import numpy as np

model_output_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_8_-30"
mask_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION"
stats_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_8_-30/Stats"
def read_json_files_into_dict(folder_path):
    json_files = {}
    # Read all .roi and .zip 
    masks = open_masks(mask_folder)
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file)) as f:
                data = json.load(f)
                file = file.replace("_cell_count.json", "")
                json_files[file] = data
                json_files[file]["animal_number"] = file.split("s")[0]
                json_files[file]["slide_number"] = file.split("s")[1].split("c")[0]
                json_files[file]["column_number"] = file.split("s")[1].split("c")[1].split("r")[0]
                json_files[file]["row_number"] = file.split("s")[1].split("c")[1].split("r")[1].split(" ")[0]
                area = 0
                for key, value in masks[file].items():
                    print(value.keys())
                    polygon = [list(zip(value["x"],value["y"]))]
                    # Calculate area of polygon
                    area += 0.5 * np.abs(np.dot(value["x"], np.roll(value["y"], 1)) - np.dot(value["y"], np.roll(value["x"], 1)))
                json_files[file]["area"] = area
                # Calculate density
                json_files[file]["density"] = json_files[file]["cell_count"] / area

    return json_files

json_files = read_json_files_into_dict(model_output_folder)
error = 0
for key, value in json_files.items():
    if value["cell_count"] == 0:
        error = error + 1
        print("Check " + key + " for errors.")
    else:
        print(value["slide_number"], value["column_number"], value["row_number"], value["cell_count"])

def reorder_dictionary(dictionary):
    return dict(sorted(dictionary.items(), key=lambda x: (x[1]["animal_number"], x[1]["row_number"], x[1]["slide_number"], x[1]["column_number"])))

sorted_dict = reorder_dictionary(json_files)

def split_dictionary(dictionary):
    animal_number_dict = {}
    for key, value in dictionary.items():
        if value["animal_number"] not in animal_number_dict:
            animal_number_dict[value["animal_number"]] = {}
        animal_number_dict[value["animal_number"]][key] = value
    return animal_number_dict

workbook = xlsxwriter.Workbook("C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_8_-30/Stats/quantification_excel_fixed_zeroes.xlsx")
worksheet = workbook.add_worksheet()

split_dict = split_dictionary(sorted_dict)
row = 1
for animal_number, images in split_dict.items():
    worksheet.write(0, 0, "image name")
    worksheet.write(0, 1, "cell count")
    worksheet.write(0, 2, "animal number")
    worksheet.write(0, 3, "slide number")
    worksheet.write(0, 4, "column number")
    worksheet.write(0, 5, "row number")
    for image_name, image_data in images.items():
        worksheet.write(row, 0, image_name)
        worksheet.write(row, 1, image_data["cell_count"])
        worksheet.write(row, 2, image_data["animal_number"])
        worksheet.write(row, 3, image_data["slide_number"])
        worksheet.write(row, 4, image_data["column_number"])
        worksheet.write(row, 5, image_data["row_number"])
        row += 1

workbook.close()

from matplotlib import pyplot as plt
# Plot density of cells in split_dict here.
for animal_number, images in split_dict.items():
    density = []
    count = []
    area = []
    for image_name, image_data in images.items():
        density.append(image_data["density"])
        count.append(image_data["cell_count"])
        area.append(image_data["area"])

    # Smooth out data.
    smooth_factor = 6
    density = [sum(density[i:i+smooth_factor])/smooth_factor for i in range(len(density)-smooth_factor-1)]
    count = [sum(count[i:i+smooth_factor])/smooth_factor for i in range(len(count)-smooth_factor-1)]
    area = [sum(area[i:i+smooth_factor])/smooth_factor for i in range(len(area)-smooth_factor-1)]
    

    # Normalize each array
    density = [x / max(density) for x in density]
    count = [x / max(count) for x in count]
    area = [x / max(area) for x in area]


    plt.plot(count)
    plt.plot(density)
    plt.plot(area)
    
    # Label the axes
    plt.xlabel("Image Number")
    plt.ylabel("Normalized Cell Count")
    plt.title("Normalized Cell Count vs. Image Number for Animal " + animal_number)
    plt.legend(["Cell Count", "Density", "Area"])

    # save plot to stats folder.
    plt.savefig(stats_folder + "/animal_" + animal_number + "_cell_count_vs_image_number.png")
    # clear plot
    plt.clf()

    # plt.show()