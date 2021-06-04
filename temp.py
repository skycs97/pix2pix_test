import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

import random
import re
import os


DATA_PATH = '/home/kmsjames/cs/data'
INPUT_IMAGE_DIR = 'Set1_input_images'
GROUND_TRUTH_IMAGE_DIR = 'Set1_ground_truth_images'
FILE_LIST_PATH = 'data_list.csv'

PATH = os.path.join(DATA_PATH, INPUT_IMAGE_DIR)

dirs = os.listdir(PATH)

print(len(dirs))

for idx, d in enumerate(dirs):
    print(idx)
    image = cv2.imread(os.path.join(PATH, d))
    image = np.transpose(image, (2,0,1))
    image = torch.from_numpy(image)
    del(image[2])        
    print(image.shape)
    break
    