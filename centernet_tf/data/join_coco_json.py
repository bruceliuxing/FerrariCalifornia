#/usr/bin/env python
#coding:utf-8

import os
import sys
import json
import random

train_split = 0.8
cur_path = os.path.dirname(os.path.abspath(__file__))

json_path_generate = "images/generate_image/generate_minsize20_maxsize240/"
json_path_online = "images/online_image/"
json_path_crawl = "images/crawl_image/"

coco_json_path = [os.path.join(cur_path, json_path, "coco_annotations/train.json") for json_path in [json_path_generate, json_path_online, json_path_crawl]]

image_all = []
annotation_all = []
categories = []
for i, json_path in enumerate(coco_json_path, 1):
    train_json = json.load(open(json_path, 'r', encoding = 'UTF-8'), encoding = 'UTF-8')
    if i == 1:
        categories.extend(train_json["categories"])
    images = train_json["images"]
    annotations = train_json["annotations"]
    image_all_len = len(image_all)
    annotation_all_len = len(annotation_all)

    for image_info in images:
        image_info["id"] += image_all_len
    for anno_info in annotations:
        anno_info["id"] += annotation_all_len
        anno_info["image_id"] += image_all_len
    image_all.extend(images)
    annotation_all.extend(annotations)

dataset_train = {"categories": categories, "images": [], "annotations": []}
dataset_val = {"categories": categories, "images": [], "annotations": []}

image_val_id = []
for image_info in image_all:
    if random.random() > train_split:
        dataset_val['images'].append(image_info)
        image_val_id.append(image_info["id"])
    else:
        dataset_train['images'].append(image_info)
for annotation_info in annotation_all:
    if annotation_info['image_id'] in image_val_id:
        dataset_val['annotations'].append(annotation_info)
    else:
        dataset_train['annotations'].append(annotation_info)
print("image_val_id:[{}]".format(len(image_val_id)))

folder = os.path.join(cur_path, 'annotations', 'nettools_generate_annos')

if not os.path.exists(folder):
    os.makedirs(folder)
train_json_name = os.path.join(folder, 'train_all.json')
with open(train_json_name,'w') as ftrain:
    json.dump(dataset_train, ftrain)

val_json_name = os.path.join(folder, 'val_all.json')
with open(val_json_name, 'w') as fval:
    json.dump(dataset_val, fval)


if __name__ == "__main__":
    pass
