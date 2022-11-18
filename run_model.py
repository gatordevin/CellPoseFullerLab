from Utils import open_images_and_masks, mask_to_countour, contour_to_mask
from matplotlib import pyplot as plt
from cellpose import models, core
from cellpose.io import logger_setup
from cellpose import plot
import numpy as np
import cv2

file_dir = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Manual Counts copy/TO TEST CODE"
save_dir = file_dir + "/model_output"

model_path = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Manual Counts copy/CellPoseTesting/split/models/CP_20221116_160410"
batch_size = 4

use_GPU = core.use_gpu()
logger_setup()

model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)

image_mask_pairs = open_images_and_masks(file_dir)
for image, mask in image_mask_pairs:
    masks, flows, styles = model.eval([image], diameter=12, flow_threshold=1.0, channels=[0,0], cellprob_threshold=-3)
    seg_mask = masks[0]

    polygons = []
    for key, value in mask.items():
        polygon = [list(zip(value["x"],value["y"]))]
        polygons.append(polygon)
    mask_image = contour_to_mask(image.shape[0:2], polygons)

    masked_segs = seg_mask * mask_image
    for pixel_val in np.unique(masked_segs)[1:]:
        single_seg_mask = np.where(masked_segs == pixel_val, True, False)
        
    
    plt.imshow(masked_segs)
    plt.show()

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