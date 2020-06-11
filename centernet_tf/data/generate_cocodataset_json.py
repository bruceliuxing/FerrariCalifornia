#!/usr/env/bin python
#coding=gbk

import os
import sys
import json
import skimage.io
import random
import cv2

cur_path = os.path.dirname(os.path.abspath(__file__))
phase_dict = {'train':1.8, 'val':0.2}
phase = "train"
split = phase_dict[phase]
dataset_train = {'categories':[], 'images':[], 'annotations':[]}
dataset_val = {'categories':[], 'images':[], 'annotations':[]}
cls_dict = {}

with open(os.path.join(cur_path, 'classes.txt')) as f:
    classes = f.read().strip().split('\n')

for i, cls in enumerate(classes, 1):
    cls = cls.split('|')[0]
    dataset_train['categories'].append({'id':i, 'name':cls, 'supercategory':cls.split('-')[0]})
    dataset_val['categories'].append({'id':i, 'name':cls, 'supercategory':cls.split('-')[0]})
    cls_dict[cls] = i

#with open(os.path.join(cur_path, "annotations/nettool_genetate_annotation.txt")) as tr:
with open(os.path.join(cur_path, sys.argv[1])) as tr:
    annos = tr.readlines()

for k, anno in enumerate(annos):
    annolist = anno.strip().split('\t')
    #im = skimage.io.imread(annolist[0])
    try:
        im = cv2.imread(annolist[0])
        if im is None:
            os.remove(annolist[0])
    except Exception as e:
        continue

    height, width, _ = im.shape
    image_info = {'file_name':annolist[0],
                              'id':k,
                              'width':width,
                              'height':height}
    anno_info = annolist[-1].split(',')
    #anno_info = annolist[1:]
    x = int(anno_info[0])
    y = int(anno_info[1])
    w = int(anno_info[2])
    h = int(anno_info[3])
    clsname = annolist[0].split('/')[-2]
    annotation_info = {
        'area': (w * h),
        'bbox':[x, y, w, h],
        'category_id':cls_dict[clsname],
        'id':k,
        'image_id':k,
        'iscrowd':0
        }
    if random.random() < split:
        dataset_train['images'].append(image_info)
        dataset_train['annotations'].append(annotation_info)
    else:
        dataset_val['images'].append(image_info)
        dataset_val['annotations'].append(annotation_info)

#folder = os.path.join(cur_path, 'annotations', 'nettools_generate_annos')
folder = os.path.abspath(os.path.dirname(sys.argv[1]))
if not os.path.exists(folder):
    os.makedirs(folder)
train_json_name = os.path.join(folder, 'train.json')
with open(train_json_name,'w') as ftrain:
    json.dump(dataset_train, ftrain)

val_json_name = os.path.join(folder, 'val.json')
with open(val_json_name, 'w') as fval:
    json.dump(dataset_val, fval)

