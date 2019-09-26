#!/home/yujs/anaconda3/bin/python

import cv2
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
from torchvision import models


def get_filelist(dir, Filelist):
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            get_filelist(newdir, Filelist)
    return Filelist


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
img_path = './256_ObjectCategories/'
img_list = get_filelist(img_path, [])


def preprocess_image(cv2im, resize_im=True):
    # mean and std list for channels (ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # processing single channel pictures
    try:
        if len(cv2im.shape) != 3:
            temp = np.zeros((cv2im.shape[0], cv2im.shape[1], 3))
            temp[:, :, 0] = cv2im
            temp[:, :, 1] = cv2im
            temp[:, :, 2] = cv2im
            cv2im = temp
            cv2im = np.transpose(cv2im, (2, 0, 1))  # Reshape
    except Exception as e:
        print(e)
        print(e.__traceback__.tb_frame.f_globals["__file__"])
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    im_as_ten = im_as_ten.cuda()
    return im_as_ten


class FeatureExtraction():
    def __init__(self, img_path):
        self.img_path = img_path
        VGG = models.vgg16(pretrained=True)
        self.pretrained_model = VGG
        self.pretrained_model.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        #print(self.pretrained_model)
        self.pretrained_model = self.pretrained_model.eval()
        self.pretrained_model = self.pretrained_model.cuda()

    def get_feature(self):
        self.pretrained_model.eval()
        img = cv2.imread(self.img_path)
        img = preprocess_image(img)
        feature = self.pretrained_model(img)
        feature = feature.data.cpu().numpy()
        feature_norm = feature/np.linalg.norm(feature)
        #print(feature_norm)
        #print(feature_norm.shape)
        return feature_norm


def save_h5(h5f, data, target):
    shape_list = list(data.shape)
    if not h5f.__contains__(target):
        shape_list[0] = None
        dataset = h5f.create_dataset(target, data=data, maxshape=tuple(shape_list), chunks=True)
        return
    else:
        dataset = h5f[target]
    len_old = dataset.shape[0]
    len_new = len_old + data.shape[0]
    shape_list[0] = len_new
    dataset.resize(tuple(shape_list))
    dataset[len_old:len_new] = data


if __name__ == '__main__':
    h5f = h5py.File('feature1.h5', 'w')
    print("Processing...")
    for i in range(len(img_list)):
        img = img_list[i]
        print(img)
        img_feature = FeatureExtraction(img).get_feature()
        save_h5(h5f, data=img_feature, target='vgg_features')
    print("Processing success!")
