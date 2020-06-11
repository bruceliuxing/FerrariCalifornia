#coding=gbk
import argparse
import os
#from tqdm import tqdm
import numpy as np
import skimage
from skimage import io, color,filters,transform
import random
import cv2
import hashlib

cur_path = os.path.dirname(os.path.abspath(__file__))
bg_image_file = os.path.join(cur_path, 'bg_img_urls.txt.bak')
bg_imgs = [url.strip() for url in open(bg_image_file).readlines()]

MAX_SIZE = 240
MAX_CROP_SIZE = (MAX_SIZE, MAX_SIZE, 3)
MAX_SCALES = 10
ROTATIONS = 4
NUM_BGIMGS = 10

#ANNOTATION_PATH = '/home/users/liuxing07/cloud_disk1/data/nettools/annotations/'
#IMAGE_PATH = '/home/users/liuxing07/cloud_disk1/data/nettools/images/generate/'
ANNOTATION_PATH = os.path.join(cur_path, "images/generate_image/generate_minsize20_maxsize{}".format(MAX_SIZE), "annotations/")
IMAGE_PATH = os.path.join(ANNOTATION_PATH, "../image")
if not os.path.exists(ANNOTATION_PATH):
    os.makedirs(ANNOTATION_PATH)
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_path", type=str, default=None, help="base sample path")
    return parser.parse_args()

def get_black_base(sample_path):
    if not os.path.exists(sample_path):
        raise Exception("Invalid sample path!")
    crop_base = {}
    if os.path.isfile(sample_path):
        brand = os.path.basename(sample_path).strip().split('_')[0]
        crop_base[brand] = io.imread(os.path.abspath(sample_path))
        return crop_base
    for root,dirs,files in os.walk(sample_path):
        if len(files) > 0:
            for perfile in files:
                if perfile.find('.') != -1:
                    brand = perfile.strip().split('_')[0]
                    crop_base[brand] = io.imread(os.path.join(root, perfile))
    return crop_base

def get_crop_rotation(crop_arr, bordervalue = 0):
    angles = random.sample(range(30, 330, 30), ROTATIONS)
    rols, cols, channels = crop_arr.shape
    rotations = []
    for angle in angles:
        M = cv2.getRotationMatrix2D((rols/2.0, cols/2.0), angle, 1)
        nW = np.int32(cols * np.abs(M[0,0]) + rols * np.abs(M[0,1]))
        nH = np.int32(cols * np.abs(M[0,1]) + rols * np.abs(M[0,0]))
        M[0, 2] += (nW / 2) - cols/2.0
        M[1, 2] += (nH / 2) - rols/2.0
        #bounder = np.vstack(([0,0,1], [cols,0,1], [0,rols,1], [cols,rols,1]))
        #new_bounder = np.dot(M, bounder.T).T
        dst = cv2.warpAffine(crop_arr, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=bordervalue)
        dst = skimage.img_as_ubyte(dst/dst.max())
        v_axis = np.argmax(np.amax(dst, axis=(0,1)))
        x, y, w, h = cv2.boundingRect(dst[:,:,v_axis])
        rotations.append(dst[y : y+h, x : x+w, :])
    return rotations

def get_crop_pyramids(crop_arr):
    sizes = [i for i in range(20, np.min(MAX_CROP_SIZE[:2]), 20)]
    if len(sizes) > MAX_SCALES:
        sizes = random.sample(sizes, MAX_SCALES)
    size_pyramids = []
    for size in sizes:
        dst = transform.resize(crop_arr, (size, size))
        dst = skimage.img_as_ubyte(dst / dst.max())
        size_pyramids.append(dst)
    return size_pyramids

def get_crop_affine(crop_arr, bordervalue = 0):
    rols, cols, channels = crop_arr.shape
    affines = []
    MA1 = np.float32([[0.7, 0.3, 0],[0, 1, 0]])
    MA2 = np.float32([[1.5,-0.5, 0],[0.2, 0.8, 0]])
    MS = [MA1, MA2]
    for M in MS:
        nW = np.int32(cols * np.abs(M[0,0]) + rols * np.abs(M[0,1]))
        nH = np.int32(cols * np.abs(M[1,0]) + rols * np.abs(M[1,1]))
        M[0, 2] += (nW - cols) /2.0
        M[1, 2] += (nH - rols) /2.0

        dst = cv2.warpAffine(crop_arr, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=bordervalue)
        dst = skimage.img_as_ubyte(dst / dst.max())
        v_axis = np.argmax(np.amax(dst, axis=(0,1)))
        x, y, w, h = cv2.boundingRect(dst[:,:,v_axis])
        affines.append(dst[y : y+h, x : x+w, :])
    return affines

def get_crop_rgb(crop_arr):
    """
    default in : black(crop_arr)
    default out: red, green, blue

    """
    rols, cols, channels = crop_arr.shape
    colors = []
    #roi_black = crop_arr.copy()
    #roi_white = np.where(crop_arr<10, 255, 0);
    #roi_red = crop_arr.copy(); roi_red[:,:,0] = 255
    roi_area = crop_arr[:,:,0] < 10
    #roi_white = 255.0 - crop_arr
    roi_red = np.zeros(crop_arr.shape); roi_red[:,:,0][roi_area] = 255
    roi_green = np.zeros(crop_arr.shape); roi_green[:,:,1][roi_area] = 255
    roi_blue = np.zeros(crop_arr.shape); roi_blue[:,:,2][roi_area] = 255
    colors.extend([roi_red, roi_green, roi_blue])
    return colors

def get_crop_flip(crop_arr):
    flip_h = crop_arr[:, ::-1, :]
    flip_v = crop_arr[::-1, :, :]
    flip = [flip_h, flip_v]
    return flip

def get_crops(crop_arr, is_black = False):
    if is_black:
        crop_arr = 255 - crop_arr
    cflip = get_crop_flip(crop_arr)
    csize = get_crop_pyramids(crop_arr)
    flip_size_crops = []
    flip_size_crops.extend(cflip)
    flip_size_crops.extend(csize)
    ctransform = []
    fst_crops = []
    for item in flip_size_crops:
        crotate = get_crop_rotation(item)
        caffine = get_crop_affine(item)
        ctransform.extend(crotate)
        ctransform.extend(caffine)
    fst_crops.extend(flip_size_crops)
    fst_crops.extend(ctransform)
    if is_black:
        return [255 - item for item in fst_crops]
    crops = []
    rgb_crops = []
    for item in fst_crops:
        crgb = get_crop_rgb(255 - item)
        rgb_crops.extend(crgb)
    crops.extend(fst_crops)
    crops.extend(rgb_crops)
    return crops

def random_paste_crop(crop_arr, is_black=False):
    bg_sample = []
    for url in random.sample(bg_imgs, NUM_BGIMGS):
        try:
            bgimage = io.imread(url)
            if not len(bgimage.shape) == 3:
                continue
            if bgimage.shape[2] == 1:
                bgimage = skimage.color.grey2rgb(bgimage)
        except:
            continue
        bg_sample.append(bgimage)
    bg_samples = [bg[:,:,:3] if bg.shape[2] > 3 else bg for bg in bg_sample]
    bgs = []
    boxes = []
    print("crop shape:{}, dtype:{}, max:{} min:{}".format(crop_arr.shape,crop_arr.dtype,crop_arr.max(), crop_arr.min()))
    for bg in bg_samples:
        if bg.shape[0] < crop_arr.shape[0] or bg.shape[1] < crop_arr.shape[1]:
            continue
        randx = 0 if (bg.shape[1] == crop_arr.shape[1]) else np.random.randint(0, bg.shape[1] - crop_arr.shape[1])
        randy = 0 if (bg.shape[0] == crop_arr.shape[0]) else np.random.randint(0, bg.shape[0] - crop_arr.shape[0])
        v_axis = np.argmax(np.amax(crop_arr, axis=(0,1)))
        color_area = crop_arr[:,:,v_axis] > 10
        if is_black:
            color_area = np.logical_not(color_area)
        bg[randy : randy+crop_arr.shape[0], randx : randx+crop_arr.shape[1], :][color_area] = crop_arr[color_area]
        bgs.append(bg)
        boxes.append([randx, randy, crop_arr.shape[1], crop_arr.shape[0]])
    return bgs, boxes

def generate_imgs(args):
    black_base_img = get_black_base(args.sample_path)
    anno_file = os.path.join(ANNOTATION_PATH, 'nettool_genetate_annotation.txt')
    fout = open(anno_file, 'a')
    for key in black_base_img.keys():
        brand = key
        black_crops = get_crops(black_base_img[key], is_black=True)
        for item in black_crops:
            bgs, boxes = random_paste_crop(item, is_black=True)
            for i in range(len(bgs)):
                bgmd5 = hashlib.md5(bgs[i].tostring()).hexdigest()
                brand_dir = os.path.join(IMAGE_PATH, brand)
                if not os.path.exists(brand_dir):
                    os.makedirs(brand_dir)
                    print("touch [{}] dir succeed!".format(brand_dir))
                bgimage_name = os.path.join(brand_dir, bgmd5 + '.jpg')
                print(bgimage_name)
                io.imsave(bgimage_name, bgs[i])
                annotation = [bgimage_name]
                annotation.append(','.join(['{}'.format(pos) for pos in boxes[i]]))
                #annotation.extend(['{}'.format(pos) for pos in boxes[i]])
                anno_res = "%s\n" % ('\t'.join(annotation))
                fout.write(anno_res)
        color_crops = get_crops(255-black_base_img[key], is_black=False)
        for item in color_crops:
            bgs, boxes = random_paste_crop(item, is_black=False)
            for i in range(len(bgs)):
                bgmd5 = hashlib.md5(bgs[i].tostring()).hexdigest()
                brand_dir = os.path.join(IMAGE_PATH, brand)
                if not os.path.exists(brand_dir):
                    os.makedirs(brand_dir)
                    print("touch [{}] dir succeed!".format(brand_dir))
                bgimage_name = os.path.join(brand_dir, bgmd5 + '.jpg')
                print(bgimage_name)
                io.imsave(bgimage_name, bgs[i])
                annotation = [bgimage_name]
                annotation.append(','.join(['{}'.format(pos) for pos in boxes[i]]))
                #annotation.extend(['{}'.format(pos) for pos in boxes[i]])
                anno_res = "%s\n" % ('\t'.join(annotation))
                fout.write(anno_res)
    fout.close()

if __name__ == '__main__':
    args = parse_cmd()
    generate_imgs(args)

