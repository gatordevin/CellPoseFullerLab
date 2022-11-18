from DatasetSplitter import open_images_and_masks
from matplotlib import pyplot as plt
from cellpose import models, core
from cellpose.io import logger_setup
from cellpose import plot

file_dir = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Manual Counts copy/TO TEST CODE"
save_dir = file_dir + "/model_output"

model_path = "C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Manual Counts copy/CellPoseTesting/split/models/CP_20221116_160410"
batch_size = 4

use_GPU = core.use_gpu()
logger_setup()

# model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path)

image_mask_pairs = open_images_and_masks(file_dir)
print(image_mask_pairs[1][1])

# images = []
# for idx, cropped_image in enumerate(generateCrops(file_dir, save_dir, crop_size, False)):
#     images.append(cropped_image)
#     if((idx+1)%batch_size==0):
#         masks, flows, styles = model.eval(images, diameter=None, flow_threshold=1.0, channels=[0,0], cellprob_threshold=0.3)
#         for iidx in range(batch_size):
#             maski = masks[iidx]
#             flowi = flows[iidx][0]

#             fig = plt.figure(figsize=(12,5))
#             plot.show_segmentation(fig, images[iidx], maski, flowi, channels=[0,0])
#             plt.tight_layout()
#             plt.show()
#         images = []