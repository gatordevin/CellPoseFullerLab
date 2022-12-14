import os
import json
import xlsxwriter
from Utils import open_masks, readAndStandardize, mask_to_json, get_gray_matter_section_polygons
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd

model_output_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_7_-20"
mask_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION"
stats_folder = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_7_-20/Stats"
file_map_excel = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_7_-20/Stats/quantification_excel.xlsx"
excel_sheet = "Sheet2"
image_name_column = "image name"

xls = pd.ExcelFile(file_map_excel)
image_info_sheet = pd.read_excel(xls, excel_sheet)
image_names = image_info_sheet.loc[:,image_name_column].tolist()

parsed_image_names = []
for image_name in image_names:
    image_name = str(image_name)
    split_image_name = image_name.split(" ")
    if(len(split_image_name)>1):
        parsed_image_names.append(split_image_name[0]+ " PM NEUN")
    else:
        parsed_image_names.append(image_name+ " PM NEUN")

def read_json_files_into_dict(folder_path):
    json_files = {}
    # Read all .roi and .zip 
    masks = open_masks(mask_folder)
    idx = 0
    for file in os.listdir(folder_path):
        if file.endswith("_cell_count.json"):
            print(str(idx) + " out of " + str(len(masks)))
            idx += 1
            with open(os.path.join(folder_path, file)) as f:
                data = json.load(f)
                file = file.replace("_cell_count.json", "")
                cell_masks_file = file + "_cell_mask.png"
                cell_mask_info_file = file + "_cell_masks_info.json"
                if cell_mask_info_file in os.listdir(folder_path):
                    with open(os.path.join(model_output_folder, cell_mask_info_file)) as f:
                        data = json.load(f)
                        json_files[file] = data
                        print("File " + file + " already processed.")
                        continue
                cell_masks = readAndStandardize(model_output_folder + "/" + cell_masks_file)

                json_files[file] = data

                contour_dicts = mask_to_json(cell_masks)
                json_files[file]["cells"] = contour_dicts
                centers = []
                diameters = []
                for contour_dict in contour_dicts:
                    center = contour_dict["center"]
                    diameter = contour_dict["equivalent_diameter"]
                    centers.append(center)
                    diameters.append(diameter)

                section_counts = {}
                for key, value in masks[file].items():
                    polygon = list(zip(value["x"],value["y"]))
                    gray_matter_section_dict = get_gray_matter_section_polygons(polygon)
                    # Check which section the center of a cell is in.
                    if(gray_matter_section_dict!=None):
                        # Plot each section with rainbow colors.
                        plt.plot(gray_matter_section_dict["left_ventral_horn"].exterior.xy[0], gray_matter_section_dict["left_ventral_horn"].exterior.xy[1], color='r')
                        plt.plot(gray_matter_section_dict["left_lateral_horn"].exterior.xy[0], gray_matter_section_dict["left_lateral_horn"].exterior.xy[1], color='orange')
                        plt.plot(gray_matter_section_dict["left_dorsal_horn"].exterior.xy[0], gray_matter_section_dict["left_dorsal_horn"].exterior.xy[1], color='y')
                        plt.plot(gray_matter_section_dict["right_dorsal_horn"].exterior.xy[0], gray_matter_section_dict["right_dorsal_horn"].exterior.xy[1], color='g')
                        plt.plot(gray_matter_section_dict["right_lateral_horn"].exterior.xy[0], gray_matter_section_dict["right_lateral_horn"].exterior.xy[1], color='b')
                        plt.plot(gray_matter_section_dict["right_ventral_horn"].exterior.xy[0], gray_matter_section_dict["right_ventral_horn"].exterior.xy[1], color='purple')
                        for section in gray_matter_section_dict:
                            section_counts[section] = 0
                            contour_array = np.array(gray_matter_section_dict[section].exterior.coords).reshape((-1,1,2)).astype(np.int32)
                            # plt.plot(contour_array[:,0,0], contour_array[:,0,1])
                            for contour in contour_dicts:
                                center = contour["center"]
                                plt.plot(center[0], center[1], "ro")
                                # Convert shapely polygon to cv2 contour.
                                if cv2.pointPolygonTest(contour_array, center, False) >= 0:
                                    contour["section"] = section
                                    section_counts[section] += 1
                                    # print("Cell is in: " + section)
                            # plt.show()

                        plt.savefig(mask_folder + "/" + file + "_sectioned.png")

                # count by section
                json_files[file]["count_by_section"] = {}
                for key, value in section_counts.items():
                    print(key + " contains " + str(value))
                    json_files[file]["count_by_section"][key] = value

                bin_size = 0.5
                bins = np.arange(0, 35, step=bin_size)
                json_files[file]["count_by_size"] = {}
                for bin in bins:
                    json_files[file]["count_by_size"][str(bin)+"-"+str(bin+bin_size)] = 0
                for diameter in diameters:
                    for bin in bins:
                        if diameter >= bin and diameter < bin+bin_size:
                            json_files[file]["count_by_size"][str(bin)+"-"+str(bin+bin_size)] += 1
                            break

                
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

                # print(json_files[file])

                # Save json file
                with open(os.path.join (folder_path, file + "_cell_masks_info.json"), "w") as f:
                    json.dump(json_files[file], f)

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

workbook = xlsxwriter.Workbook("C:/Users/gator/OneDrive - University of Florida/10x images for quantification/PM NEUN FOR QUANTIFICATION/model_output_12_7_-20/Stats/quantification_excel_labeled_area_w_empties.xlsx")
worksheet = workbook.add_worksheet()

row = 1
worksheet.write(0, 0, "image name")
worksheet.write(0, 1, "cell count")
worksheet.write(0, 2, "animal number")
worksheet.write(0, 3, "slide number")
worksheet.write(0, 4, "column number")
worksheet.write(0, 5, "row number")
column = 6

for key, value in sorted_dict[list(sorted_dict.keys())[0]]["count_by_size"].items():
    worksheet.write(0, column, "Diameter " + key)
    column += 1
for key, value in sorted_dict[list(sorted_dict.keys())[0]]["count_by_section"].items():
    worksheet.write(0, column, "Section " + key)
    column += 1

for image_name in parsed_image_names:
    worksheet.write(row, 0, image_name)
    if(image_name in sorted_dict.keys()):
        image_data = sorted_dict[image_name]
        worksheet.write(row, 1, image_data["cell_count"])
        worksheet.write(row, 2, image_data["animal_number"])
        worksheet.write(row, 3, image_data["slide_number"])
        worksheet.write(row, 4, image_data["column_number"])
        worksheet.write(row, 5, image_data["row_number"])
        column = 6
        for key, value in image_data["count_by_size"].items():
            worksheet.write(row, column, value)
            column += 1
        for key, value in image_data["count_by_section"].items():
            worksheet.write(row, column, value)
            column += 1
    row += 1

workbook.close()


# split_dict = split_dictionary(sorted_dict)
# row = 1
# for animal_number, images in split_dict.items():
#     worksheet.write(0, 0, "image name")
#     worksheet.write(0, 1, "cell count")
#     worksheet.write(0, 2, "animal number")
#     worksheet.write(0, 3, "slide number")
#     worksheet.write(0, 4, "column number")
#     worksheet.write(0, 5, "row number")
#     column = 6
#     for key, value in images[list(images.keys())[0]]["count_by_size"].items():
#         worksheet.write(0, column, "Diameter " + key)
#         column += 1
#     for key, value in images[list(images.keys())[0]]["count_by_section"].items():
#         worksheet.write(0, column, "Section " + key)
#         column += 1

#     for image_name in parsed_image_names:
#         worksheet.write(row, 0, image_name)
#         if(image_name in images.keys()):
#             image_data = images[image_name]
#             worksheet.write(row, 1, image_data["cell_count"])
#             worksheet.write(row, 2, image_data["animal_number"])
#             worksheet.write(row, 3, image_data["slide_number"])
#             worksheet.write(row, 4, image_data["column_number"])
#             worksheet.write(row, 5, image_data["row_number"])
#             column = 6
#             for key, value in image_data["count_by_size"].items():
#                 worksheet.write(row, column, value)
#                 column += 1
#             for key, value in image_data["count_by_section"].items():
#                 worksheet.write(row, column, value)
#                 column += 1
#         row += 1

# workbook.close()

# from matplotlib import pyplot as plt
# Plot density of cells in split_dict here.
# for animal_number, images in split_dict.items():
#     density = []
#     count = []
#     area = []
#     for image_name, image_data in images.items():
#         density.append(image_data["density"])
#         count.append(image_data["cell_count"])
#         area.append(image_data["area"])

#     # Smooth out data.
#     smooth_factor = 6
#     density = [sum(density[i:i+smooth_factor])/smooth_factor for i in range(len(density)-smooth_factor-1)]
#     count = [sum(count[i:i+smooth_factor])/smooth_factor for i in range(len(count)-smooth_factor-1)]
#     area = [sum(area[i:i+smooth_factor])/smooth_factor for i in range(len(area)-smooth_factor-1)]
    

#     # Normalize each array
#     density = [x / max(density) for x in density]
#     count = [x / max(count) for x in count]
#     area = [x / max(area) for x in area]


#     plt.plot(count)
#     plt.plot(density)
#     plt.plot(area)
    
#     # Label the axes
#     plt.xlabel("Image Number")
#     plt.ylabel("Normalized Cell Count")
#     plt.title("Normalized Cell Count vs. Image Number for Animal " + animal_number)
#     plt.legend(["Cell Count", "Density", "Area"])

#     # save plot to stats folder.
#     plt.savefig(stats_folder + "/animal_" + animal_number + "_cell_count_vs_image_number.png")
#     # clear plot
#     plt.clf()

    # plt.show()