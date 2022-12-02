import os
from Utils import readAndStandardize, dots_image_to_density, normalizeImages
from matplotlib import pyplot as plt
import numpy as np

model_output_path = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Coding/TO TEST CODE/model_output_12_8_-30"

dot_images = []
for file_name in os.listdir(model_output_path):
    if(file_name.endswith("_cell_dots.png")):
        image = readAndStandardize(model_output_path + "/" + file_name)
        dot_images.append(image)

density_images = []
for dot_image in dot_images:
    density_img = dots_image_to_density(dot_image, 35)
    density_images.append(density_img)

minMaxedImages = normalizeImages(density_images, method="minmax")
normalizedImages = normalizeImages(density_images, method="normal")

fig, axs = plt.subplots(3,3)
fig.suptitle('Vertically stacked subplots')

axs[0][0].imshow(density_images[0])
axs[1][0].imshow(normalizedImages[0], vmin = -2, vmax = 10.0)
axs[2][0].imshow(minMaxedImages[0], vmin = 0, vmax = 1)

axs[0][1].imshow(density_images[1])
axs[1][1].imshow(normalizedImages[1], vmin = -2, vmax = 10.0)
axs[2][1].imshow(minMaxedImages[1], vmin = 0, vmax = 1)

axs[0][2].imshow(density_images[2])
axs[1][2].imshow(normalizedImages[2], vmin = -2, vmax = 10.0)
axs[2][2].imshow(minMaxedImages[2], vmin = 0, vmax = 1)

plt.show()