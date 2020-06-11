#/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json
import random
import cv2

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

ANNOTATION_PATH = os.path.join(CUR_PATH, "annotations")
CLASSES_PATH = os.path.join(CUR_PATH, "classes.txt")

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--annotation_path', default='data/annotations', help='标注精灵助手标注结果(json形式导出)目录dir')
        self.parser.add_argument('--classes', default='classes.txt', help='标注数据类别名称文件地址')
        self.parser.add_argument('--out_anno_path', default="annotations", help='输出coco格式json文件')
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        opt.annotation_path = os.path.join(CUR_PATH, opt.annotation_path)
        opt.classes = os.path.join(CUR_PATH, opt.classes)
        opt.out_anno_path = os.path.join(opt.annotation_path, '../')
        if not os.path.exists(opt.annotation_path):
            print("{} not exists!".format(opt.annotation_path))
        if not os.path.exists(opt.classes):
            print("{} not exists!".format(opt.classes))
        return opt

class convert(object):
    def __init__(self, opt):
        self.phase_dict = {"train":1.1, "val":0.1}
        self.phase = "train"
        self.phase_size = self.phase_dict[self.phase]
        self.opt = opt
        self.train_dataset = {'categories':[], 'images':[], 'annotations':[]}
        self.val_dataset = {'categories':[], 'images':[], 'annotations':[]}
        self.cls_dict = {}
        self.cls_zh_en = {}
        #self.pre_process(self.opt.classes)

    def pre_process(self, classes_file):
        with open(classes_file) as f:
            classes = f.read().strip().split('\n')
        for i, cls in enumerate(classes, 1):
            en_cls, zh_cls = cls.split('|')
            self.cls_zh_en[zh_cls] = en_cls
            cls = en_cls
            super_category = cls.split('-')[0]
            self.train_dataset['categories'].append({'id':i, 'name':cls, 'supercategory':super_category})
            self.val_dataset['categories'].append({'id':i, 'name':cls, 'supercategory':super_category})
            self.cls_dict[cls] = i

    def get_json_file(self, annotation_path):
        if not os.path.isdir(annotation_path):
            print("this path [{}] is not directory!".format(annotation_path))
        filelist = []
        for root, dirs, files in os.walk(annotation_path):
            if len(files) > 0:
                for perfile in files:
                    if perfile.find("DS_Store") != -1:
                        try:
                            print(perfile)
                            os.remove(os.path.join(root, perfile))
                        except Exception as e:
                            print(e)
                        continue
                    if perfile.find(".json") != -1:
                        filelist.append(os.path.abspath(os.path.join(root, perfile)))
        if len(filelist) < 1:
            print("get empty json files!")
            return None
        return filelist

    def get_json_info(self, perfile):
        #if len(filelist) < 1:
        #    print("empty input! [{}]".format(filelist))
        #print("length of filelist is:[{}]".format(len(filelist)))
        #for perfile filelist:
        if not os.path.isfile(perfile):
            print("{} is not a file!".format(perfile))
            return None
        json_info = json.loads(open(perfile).read().strip(), encoding="UTF-8")
        imgpath = os.path.join(self.opt.annotation_path, "../image", json_info["path"].strip().split("online_image")[-1][1:])#.encode("utf-8")
        if " logo" in imgpath:
            imgpath = imgpath.replace(" logo", "")
        if (not json_info["labeled"]) or (len(json_info["outputs"]["object"]) < 1) or (json_info["size"]["depth"] != 3):
            try:
                os.remove(imgpath)
                return None
            except Exception as e:
                print(e)
                return "exception ocurred!"
        image_arr = cv2.imread(imgpath)
        if image_arr is None:
            try:
                os.remove(imgpath)
                return None
            except Exception as e:
                print(e)
                return "exception ocurred!"
        object_box = []
        width = int(json_info["size"]["width"])
        height = int(json_info["size"]["height"])
        depth = int(json_info["size"]["depth"])
        image_info = {"file_name":imgpath, "width":width, "height":height, "depth":depth}
        for box in json_info["outputs"]["object"]:
            name = box["name"]
            x = int(box["bndbox"]["xmin"])
            y = int(box["bndbox"]["ymin"])
            box_width = int(box["bndbox"]["xmax"] - x)
            box_height = int(box["bndbox"]["ymax"] - y)
            object_box.append({"name":name, "x":x, "y":y, "width":box_width, "height":box_height})
        return {"image_info":image_info, "object_box":object_box}

    def make_coco_data(self, filelist):
        #for k, perfile in enumerate(filelist, 1):
        img_idx = 0
        box_idx = 0
        for perfile in filelist:
            json_info = self.get_json_info(perfile)
            if json_info is None:
                os.remove(perfile)
            if "exception" in json_info:
                continue
            else:
                image_info = json_info["image_info"]
                object_box = json_info["object_box"]
                if (image_info["depth"] == 3) and (len(object_box) > 0):
                    img_idx += 1
                    image_info["id"] = img_idx
                    boxes = []
                    for box in object_box:
                        box_idx += 1
                        box_info = {"area": box["width"] * box["height"],
                                    "bbox": [box["x"], box["y"], box["width"], box["height"]],
                                    "category_id": self.cls_dict[self.cls_zh_en[box["name"]]],
                                    "id": box_idx,
                                    "image_id": img_idx,
                                    "iscrowd":0}
                        boxes.append(box_info)
                    if random.random() < self.phase_size:
                        self.train_dataset["images"].append(image_info)
                        self.train_dataset["annotations"].extend(boxes)
                    else:
                        self.val_dataset["images"].append(image_info)
                        self.val_dataset["annotations"].extend(boxes)

    def generate_coco_file(self):
        coco_anno_path = os.path.join(opt.out_anno_path, "coco_annotations")
        if not os.path.exists(coco_anno_path):
            os.makedirs(coco_anno_path)
        with open(os.path.join(coco_anno_path, "train.json"), "w", encoding="UTF-8") as ftrain:
            json.dump(self.train_dataset, ftrain)
        with open(os.path.join(coco_anno_path, "val.json"), "w", encoding="UTF-8") as fval:
            json.dump(self.val_dataset, fval)

    def join(self):
        self.pre_process(self.opt.classes)
        filelist = self.get_json_file(self.opt.annotation_path)
        self.make_coco_data(filelist)
        self.generate_coco_file()

if __name__ == "__main__":
    opt = opts().parse("--annotation_path images/online_image/annotation --classes classes.txt".split(" "))
    print(opt)
    convert_data = convert(opt)
    convert_data.join()
















