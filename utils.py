import os
import glob
import cv2
import numpy as np


def read_img(fname):
    img = cv2.imread(fname)
    return img


def read_mat(fname):
    return np.loadtxt(fname)


def read_data(dirname):
    K_path = os.path.join(dirname, 'K.txt')
    K = read_mat(K_path)
    img_list = glob.glob(os.path.join(dirname, '*.jpg'))
    imgs = [read_img(fname) for fname in img_list]
    return imgs, K
