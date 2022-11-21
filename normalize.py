from cellpose.transforms import normalize_img
from skimage.io import imread
from matplotlib import pyplot as plt
import os

folder_path = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Manual Counts copy/CellPoseTesting/"
for file_name in os.listdir(folder_path):
    if(".tif" in file_name):
        image_path = folder_path + file_name
        fig, axes = plt.subplots(1,2)
        fig.set_size_inches(12,6)
        img = imread(image_path)[1000:1600, 1000:1600]
        axes[0].imshow(img)
        normal_img = normalize_img(img)
        axes[1].imshow(normal_img)
        plt.show()