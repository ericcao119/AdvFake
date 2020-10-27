import cv2
import os
import sys
image_num = 1
for folder in os.listdir('facades/'):
    for file in os.listdir('facades/{}/'.format(folder)):
        img = cv2.imread('facades/{}/{}'.format(folder,file))
        print('facades/{}/{}'.format(folder,file))
        h, w = img.shape[0], img.shape[1]
        half_w = w//2
        half_h = h//2
        train = img[:,:half_w]
        sol = img[:,half_w:]
        cv2.imwrite('image/{}.jpg'.format(image_num), train)
        cv2.imwrite('solution/{}.jpg'.format(image_num), sol)
        image_num +=1
