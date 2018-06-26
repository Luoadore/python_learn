# coding： utf-8

"""
Image resize ops of tensorflow.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

image = tf.gfile.FastGFile('D:\OD\myCat\Cat_7185.JPG', 'rb').read()

# raw image
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())
    plt.imshow(img_data.eval())
    plt.show()

# resize image
with tf.Session() as sess:

    resized = tf.image.resize_images(img_data, [300, 300], method=0) # 双线性插值法（Bilinear interpolation）
    # resized = tf.image.resize_images(img_data, [300, 300], method=1) # 最近邻居法（Nearest neighbor interpolation）
    # resized = tf.image.resize_images(img_data, [300, 300], method=2) # 双三次插值法（Bicubic interpolation）
    # resized = tf.image.resize_images(img_data, [300, 300], method=3) # 面积插值法（Area interpolation）

    # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
    print("Digital type: ", resized.dtype)
    print("Digital shape: ", resized.get_shape())
    cat = np.asarray(resized.eval(), dtype='uint8')
    # tf.image.convert_image_dtype(rgb_image, tf.float32)
    plt.imshow(cat)
    plt.show()