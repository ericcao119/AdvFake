#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_face(template_path: str, base_image_path: str):

    img_rgb = cv2.imread(base_image_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)

    height, width = template.shape[::]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
    plt.imshow(res, cmap='gray')

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    print(min_val)
    print(max_val)

    top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 2) 

    plt.imshow(img_rgb)
    plt.show()

# %%
