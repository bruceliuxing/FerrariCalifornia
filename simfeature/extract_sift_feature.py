#!/usr/bin/env python
#coding=utf-8

import os
import sys
import numpy as np
import cv2
import tensorflow as tf
#import skimage

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
mendian_imgs_path = "./imgs/"

# cyvlfeat_path = "${ANACONDA_PATH}/envs/tf2/lib/python3.6/site-packages/cyvlfeat/"

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def list_images(basePath, contains=None):
    return list_files(basePath, validExts=image_types, contains=contains)

def list_files(basePath, validExts=None, contains=None):
    print("data path:[{}]".format(basePath))
    for root, dirs, files in os.walk(basePath, followlinks=True):
        files[:] = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for filename in files:
            # contains is a string variable 
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.find('.'):].lower()
            if validExts is None or ext.endswith(validExts):
                imagepath = os.path.join(root, filename)
                yield imagepath

def get_feature(image, feature="sift"):
    """
    feature: sift / surf
    """
    dims = len(image.shape)
    if dims != 2:
        channel = image.shape[-1]
        if channel == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channel == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            print("wrong image channel format![image shape:{}]".format(image.shape))

    if feature == "surf":
        feature = cv2.xfeatures2d_SURF.create() #best feature number
    else:
        feature = cv2.xfeatures2d_SIFT.create() #100 
    kp, des = feature.detectAndCompute(image, None)
    return kp, des

def test_img():
    img = cv2.imread('./logo_test_data/vari_pos/UCbrowser_1.jpg')
    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    import pdb;pdb.set_trace()
    kp, res = get_feature(gray)

def feature_to_file(image_path, feature='sift'):
    imagepaths = list(list_images(image_path))
    feature_dir = os.path.join(CUR_PATH, "feature", feature)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    fileout = os.path.join(feature_dir, '{}.txt'.format(feature))
    fout = open(fileout, 'w', encoding="utf-8")
    for imagepath in imagepaths:
        img = cv2.imread(imagepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape
        image_shape_str = '%s;%s'%(shape)

        kp, res = get_feature(gray)
        res_shape = res.shape
        res_shape_str = '%s;%s'%(res_shape)
        res_str = str(res.tostring())
        
        kp_num = len(kp)
        kp_content = []
        for item in kp:
            x,y = item.pt
            size = item.size
            angle = item.angle
            response = item.response
            octave = item.octave
            class_id = item.class_id
            kpcontent = [str(i) for i in [x,y,size,angle,response,octave,class_id]]
            kp_content.append(';'.join(kpcontent))
        kp_str = '|'.join(kp_content)

        print('\t'.join([imagepath, image_shape_str, res_shape_str]))

        txt_res = '\t'.join([imagepath, image_shape_str, res_shape_str, kp_str, res_str])
        fout.write('%s\n'%txt_res)
    fout.close()

def get_image(imageurl, size):
    image = cv2.imread(imageurl)
    if image is None:
        return None
    dims = len(image.shape)
    if dims == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif dims == 3:
        channel = image.shape[-1]
        if channel == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif not channel == 3:
            print("wrong image channel format![image shape:{}]".format(image.shape))
            return None
    image = image[:,:,::-1]
    image = np.array(cv2.resize(image, (size, size)), dtype="float") / 255.0
    return image
    #if imagepath.startswith("http"):
    #img = skimage.io.imread(imageurl)
    #if img is None:
    #    return None
    #dims = len(img.shape)
    #if dims == 2:
    #    img = skimage.color.gray2rgb(img)
    #elif dims == 3:
    #    channel = img.shape[-1]
    #    if channel == 4:
    #        img = skimage.color.rgba2rgb(img)
    #    elif not channel == 3:
    #        print("invalid image format![{}]".format(img.shape))
    #        return None
    #img = skimage.transform.resize(img, (image_size, image_size))
    #img = np.expand_dims(img, axis=0)
    #return img

net_imgsize = {"resnet50":224, "resnet101":224}

def get_model(feature):
    image_shape = (net_imgsize[feature], net_imgsize[feature], 3)
    layer_num = feature.split("resnet")[-1]
    func_name = "resnet{}.ResNet{}".format(layer_num, layer_num)
    model = getattr(tf.keras.applications, func_name)(input_shape=image_shape, weights="imagenet", include_top=True)
    return model

def get_sift_str(image, feature_num):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    num, vecdims = des.shape
    if num > feature_num:
        new_des = des[:feature_num, :].flatten()
    else:
        new_des = np.zeros((feature_num, vecdims))
        new_des[0:num, :] = des
        new_des = new_des.flatten()
    return '\t'.join(['{:8f}'.format(float(item)) for item in list(new_des)])

def get_net_feature(imgpath, feature="resnet50-sift10"):
    net_feature = feature.split('-')[0]
    sift_num = int(feature.split('-')[-1].split('sift')[-1])
    imgpath = os.path.abspath(imgpath)
    if os.path.isdir(imgpath):
        imagepaths = list(list_images(image_path))
    elif os.path.isfile(imgpath):
        imagepaths = [item.split(' ')[0].strip() for item in open(imgpath, encoding="UTF-8").readlines()]
    else:
        print("invalid imgpath!")
        return None
    feature_dir = os.path.join(CUR_PATH, "feature", feature)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    if "query" in imgpath:
        fileout = os.path.join(feature_dir, '{}.txt.query'.format(feature))
    else:
        fileout = os.path.join(feature_dir, '{}.txt'.format(feature))
    fout = open(fileout, 'w', encoding="UTF-8")
    from tqdm import tqdm
    image_shape = (net_imgsize[net_feature], net_imgsize[net_feature], 3)
    model = tf.keras.applications.resnet50.ResNet50(input_shape=image_shape, weights="imagenet", include_top=True)
    #model = get_model(net_feature)
   
    sift_str_dict = {}
    sift_filename = "feature/sift/imgpath.txt.sift-num10"
    for line in open(sift_filename).readlines():
        line = line.strip().split("\t")
        sift_str_dict[line[0]] = '\t'.join(line[1:])

    base_img_url = "${SERVER_ADDR}"
    for imagepath in tqdm(imagepaths):
        try:
            print(imagepath)
            img = get_image(imagepath, image_shape[0])
            if img is None:
                continue
            prediction = model.predict(np.expand_dims(img, 0))
            netvec_str = '\t'.join(["{:8f}".format(item) for item in list(prediction[0])])
            sift_str = sift_str_dict[imagepath]#get_sift_str(img, sift_num)
            result = base_img_url + imagepath.split("public")[-1] + '\t' + netvec_str + '\t' + sift_str
            fout.write('%s\n'%result)
        except Exception as e:
            print(e)
            pass
        
def main():
    #test_img()
    imgpath_file =  os.path.join(CUR_PATH, "imgpath.txt")
    get_net_feature(imgpath_file)
    image_file_query = imgpath_file + ".query"
    get_net_feature(image_file_query)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    main()

