# coding: utf-8
"""
The geometric transformation of images, such as scaling, rotation and flip, plays an important role in image processing.
"""

from PIL import Image

im = Image.open('test.jpg')
m, n = im.size
im.show()

# scaling, 主要是缩小
im_resized = im.resize((128, 128))
im_resized.show()

# rotate, 逆时针
im_rotate = im.rotate(45)

# flip
out_1 = im.transpose(Image.FLIP_LEFT_RIGHT)
out_2 = im.transpose(Image.FLIP_TOP_BOTTOM)
out_3 = im.transpose(Image.ROTATE_90)
out_4 = im.transpose(Image.ROTATE_180)
out_5 = im.transpose(Image.ROTATE_270)

import cv2
im = cv2.imread('test.jpg')
enlarge = cv2.resize(im, (0, 0), fx = 1.2, fy = 1.2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('src', im)
cv2.imshow('enlarge', enlarge)
cv2.waitKey(0)