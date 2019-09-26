#!/usr/bin/env python

import os
import numpy as np
import torch
import torch.nn as nn
import h5py
import Extraction_VGG_pretrain as Extraction
import matplotlib.pyplot as plt


def getDataFromH5py(fileName, target):
    h5f = h5py.File(fileName)
    if not h5f.__contains__(target):
        res = []
    else:
        res = h5f[target][:]
    h5f.close()
    return res


def getmaPSize(datapath):
    mAP_num = []
    for _, dirs, _ in os.walk(datapath):
        dirs.sort()
        # print(dirs)
        for j in range(257):
            d = os.listdir(os.path.join(datapath, dirs[j]))
            d.sort()
            # print(d)
            mAP_num.append(len(d))
        break
    return mAP_num


def getmaPList(datapath):
    mAP_list = []
    for _, dirs, _ in os.walk(datapath):
        dirs.sort()
        #print(dirs)
        for j in range(257):
            d = os.listdir(os.path.join(datapath, dirs[j]))
            d.sort()
            # print(d)
            path = os.path.join(datapath, dirs[j], d[0])
            mAP_list.append(path)
        break
    return mAP_list


feature = getDataFromH5py('feature.h5', 'vgg_features')
img_path = './256_ObjectCategories/'
img_list = Extraction.get_filelist(img_path, [])
# query_img = './picture1.jpg'
mAP_list = getmaPList(img_path)
mAP_size = getmaPSize(img_path)


def IsSameClass(path1, path2):
    if path1[23:26] == path2[23:26]:
        return True
    else:
        return False


class ImageQuery:
    def __init__(self, feature, query_img, img_list, query_num):
        self.feature = feature
        self.query_img = query_img
        self.img_list = img_list
        self.query_num = query_num
        self.result = {}

    def iter_database(self):
        query_img_feature = Extraction.FeatureExtraction(self.query_img).get_feature()
        query_img_feature = np.array(query_img_feature)
        query_img_feature = torch.from_numpy(query_img_feature)
        # print(query_img_feature)
        score = {}
        # print(feature.shape)
        print("Searching...")
        for i in range(len(feature)):
            # if database contains query-image
            img = self.img_list[i]
            # print("origin: "+img)
            img_feature = feature[i]
            img_feature = np.array(img_feature)
            img_feature = torch.from_numpy(img_feature)
            output = torch.dist(img_feature, query_img_feature, p=1)
            if output == 0:
                continue
            # print(output)
            score[img] = abs(output)
        print("Search Finished")
        sorted_score = sorted(score.items(), key=lambda score: score[1], reverse=False)
        self.result = sorted_score
        count = 0
        print("10 most relevant pictures are as below: ")
        for key in sorted_score:
            print(key[0])
            # image = plt.imread(key[0])
            # plt.imshow(image)
            # plt.show()
            # plt.savefig('./result/picture')
            count = count + 1
            if count == 10:
                break

    def AP_evaluation(self):
        AP = 0.0
        img = self.query_img
        self.iter_database()
        result = self.result
        j = 0
        count = 0
        total_ap = 0.0
        for key in result:
            idx = key[0]
            if IsSameClass(img, idx):
                count = count + 1
                #print("count = " + str(count))
                total_ap += count / (j + 1)
                if count == self.query_num - 1:  # exclude itself
                    break
            j = j + 1
        total_ap /= count
        AP += total_ap
        print("AP=" + str(AP))
        return AP


if __name__ == '__main__':
    mAP = 0.0
    for i in range(len(mAP_list)):
        myClass = ImageQuery(feature, mAP_list[i], img_list, mAP_size[i])
        mAP += myClass.AP_evaluation()
    mAP /= len(mAP_list)
    print(mAP)
'''
#test mode
if __name__ == '__main__':
    myClass = ImageQuery(feature, query_img, img_list)
    myClass.iter_database()
'''
