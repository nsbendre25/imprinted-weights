import numpy as np
import glob
import os
import cv2

Training=False

if(Training==True):
    path = "./mnist_png/training/"
    label = sorted(os.listdir(path))

    for i in range(0,len(label)):
        images = [cv2.imread(file, 0) for file in sorted(glob.glob(path+label[i]+'/*.png'))]
        np.save('./training_images_npy/images_{}.npy'.format(i), images)
        labels = np.repeat(label[i], len(images))
        np.save('./training_labels_npy/labels_{}.npy'.format(i), labels)
else:
    path = "./mnist_png/testing/"
    label = sorted(os.listdir(path))

    for i in range(0,len(label)):
        images = [cv2.imread(file, 0) for file in sorted(glob.glob(path+label[i]+'/*.png'))]
        np.save('./testing_images_npy/images_{}.npy'.format(i), images)
        labels = np.repeat(label[i], len(images))
        np.save('./testing_labels_npy/labels_{}.npy'.format(i), labels)
