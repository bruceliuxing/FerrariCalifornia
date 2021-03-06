#! /usr/bin/env python
#coding=utf-8
import os
import sys
import time

import tensorflow as tf
import skimage
import cv2
import requests
import numpy as np
from tqdm import tqdm

from models.resnet import centernet
from common import *
from models.resnet import decode
#from generators.utils import get_affine_transform, affine_transform

cur_path = os.path.dirname(os.path.abspath(__file__))
image_size = 256
threshold = 0.5
classes = ["maotai", "xifeng"]
def main(url_filein):
    #model_path = os.path.join(cur_path, "checkpoints/2020-05-26-hourglass/model_prediction_final.h5")
    #model_path = os.path.join(cur_path, "checkpoints/2020-05-27-resnet50/model_prediction_final.h5")
    model_path = "checkpoints/2020-05-26-hourglass/hourglass_256_debugmodel_prediction.h5"
    model = tf.keras.models.load_model(model_path)

    fileout = url_filein + ".wine_tf_hourglass256"
    fout = open(fileout, 'w')

    for ik, url in enumerate(tqdm(open(url_filein).readlines())):
        try:
#            url = "http://fc1tn.baidu.com/it/u=1217667032,1187382556&fm=202"
            image = np.asarray(bytearray(requests.get(url).content), dtype="uint8");t1=time.time()
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            img_w = image.shape[1]
            img_h = image.shape[0]
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
            s = max(image.shape[0], image.shape[1]) * 1.0
            trans_input = get_affine_transform(c, s, (image_size, image_size))
            image = cv2.warpAffine(image, trans_input, (image_size, image_size), flags=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            input = image / 255.0
            #input = input[np.newaxis,:]
            input = np.expand_dims(input, axis=0)
            output = model(input)
            outputs = decode(hm=output[0], wh=output[1], reg=output[2], num_classes=2)[0][0]
            #print(time.time()-t1)
            if outputs[-2] < threshold:
                continue
            result = [str(i) for i in [url.strip(), classes[int(outputs[-1])], float(outputs[-2])]]
            print('\t'.join(result))
            fout.write("%s\n" % ('\t'.join(result)))
        except Exception as e:
            continue
def test_debugmodel():
    in_size = 256
    arch_net = 'hourglass' # 2020-06-04 modified hourglass network
    model_path = sys.argv[1]
    #model_path = os.path.join(cur_path, "checkpoints/2020-05-26-hourglass/save_model.h5")
    #model_path = os.path.join(cur_path, "checkpoints/2020-05-27-resnet50/save_model.h5")
    #model_path = os.path.join(cur_path, "checkpoints/2020-06-08/save_model.h5")
    from models.resnet import centernet
    trainmodel,deploymodel,debugmodel = centernet(num_classes=4, backbone=arch_net, input_size=in_size)
    debugmodel.load_weights(model_path, by_name=True, skip_mismatch=True)
#    debugmodel = tf.keras.models.load_model("hourglass_384_debugmodel_prediction.h5")

#    debugmodel.summary()
#    img = cv2.imread("test_imgs/wine_maotai.jpg")
    img = cv2.imread(sys.argv[2])
    model_name = sys.argv[2].split('/')[-1].split('_')[0]
    img_w = img.shape[1]
    img_h = img.shape[0]
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0
    image = preprocess_image(img, c, s, in_size, in_size)
    image = image / 255.0
    image = image[np.newaxis,:]
    outputs = debugmodel.predict(image)
    detections = decode(hm=outputs[0], wh=outputs[1], reg=outputs[2], num_classes=2)[0][0]
    offset = (s-img_h)/2
    bbox=detections
    x1 = (int)(bbox[0] * 4 / image_size * s)
    y1 = (int)((bbox[1] * 4 / image_size * s)-offset)
    x2 = (int)(bbox[2] * 4 / image_size * s)
    y2 = (int)((bbox[3] * 4 / image_size * s)-offset)
    result = np.array([x1, y1, x2-x1, y2-y1, bbox[-2], bbox[-1]])

    print(result)

    debugmodel.save(os.path.join(os.path.dirname(model_path), "{}_{}_{}_debugmodel_prediction.h5".format(model_name, arch_net, in_size)), include_optimizer=False)

def test_tflite():
    in_size = 384
    arch_net = 'hourglass'
#    #model_path = os.path.join(cur_path, "checkpoints/2020-05-26-hourglass/save_model.h5")
#    model_path = os.path.join(cur_path, "checkpoints/2020-05-27-resnet50/save_model.h5")
#    from models.resnet import centernet
#    trainmodel,deploymodel,debugmodel = centernet(num_classes=2, backbone=arch_net, input_size=in_size)
#    debugmodel.load_weights(model_path, by_name=True, skip_mismatch=True)

#    """tf 2.2 api"""
#    debugmodel = tf.keras.models.load_model("checkpoints/2020-05-26-hourglass/save_model.h5", compile=False)
#    converter = tf.lite.TFLiteConverter.from_keras_model(debugmodel)
#    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#    converter.target_spec.supported_types = [tf.float16]
#    tflite_quant_model = converter.convert()
#    tflite_model_fp16_file = "converted_model_f16.tflite"
#    open(converted_model_f16.tflite, "wb").write(tflite_quantized_model)

#    """tf 2.2 compat.v1 api"""
#    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("checkpoints/2020-05-26-hourglass/hourglass_384_debugmodel_prediction.h5")
#    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#    tflite_quant_model = converter.convert()
    tflite_model_fp8_file = "converted_model_f8.tflite"
#    open(tflite_model_fp8_file, "wb").write(tflite_quant_model)


    img = cv2.imread("test_imgs/wine_maotai.jpg")
    img_w = img.shape[1]
    img_h = img.shape[0]
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    s = max(image.shape[0], image.shape[1]) * 1.0
    image = preprocess_image(img, c, s, in_size, in_size)
    image = image / 255.0
    image = image[np.newaxis,:].astype(np.float32)

#    import pdb;pdb.set_trace()
    libs = ["libcudart.so.10.1", "libcublas.so.10", "libcufft.so.10", "libcurand.so.10", "libcusolver.so.10", "libcusparse.so.10", "libcudnn.so.7"]
    interpreter_fp8 = tf.lite.Interpreter(model_path=str(tflite_model_fp8_file), experimental_delegates=[tf.lite.experimental.load_delegate(item) for item in libs])
    interpreter_fp8.allocate_tensors()
    input_index = interpreter_fp8.get_input_details()[0]["index"]
    output_list = interpreter_fp8.get_output_details()
    y_index = [item["index"] for item in output_list]
    for i in range(10):
        interpreter_fp8.set_tensor(input_index, image)
        t1 = time.time()
        interpreter_fp8.invoke()
        outputs = [interpreter_fp8.get_tensor(index) for index in y_index]
        detections = decode(hm=outputs[0], wh=outputs[1], reg=outputs[2], num_classes=2)
        print(detections[0][0], "{}".format(time.time() - t1))




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    img_urls = os.path.join(cur_path, "evaluate/haudit_201911_imgurls")
    #main(img_urls)
    test_debugmodel()
#    test_tflite()



