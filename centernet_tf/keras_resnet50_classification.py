#!/usr/bin/env python
#coding:utf-8

import os
import sys

#from sklearn.preprocessing import LabelBinarlizer
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report

import sklearn
#import keras
import numpy as np
import argparse
import random
import pickle
import cv2
import json
import skimage

import tensorflow as tf
from tensorflow import keras

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CUR_PATH, 'data')

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
image_size = 224
classes_num = 3

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


def get_data(path):
    print("---------开始读取数据---------")
    data = []
    labels = []
    valid_imagepath = []
    imagepaths = sorted(list(list_images(path)))
    print(len(imagepaths))
    random.seed(42)
    random.shuffle(imagepaths)
    count = 1;
    for imagepath in imagepaths:
        count += 1
        if count > 1000000:
            break
        image = cv2.imread(imagepath)
        if image is None:
            os.remove(imagepath)
            continue
        else:
            image = image[:,:,::-1]
        image = cv2.resize(image, (image_size, image_size))
        validpath = imagepath.split('vehicle_watch_classify')[-1]
        if 'vehicle' in validpath:
            label = 1
        elif 'watch' in validpath:
            label = 2
        elif 'other' in validpath:
            label = 0
        else:
            print('wrong imagepath:[{}]'.format(imagepath))
            continue
        data.append(image)
        labels.append(label)

        image_lflip = image[:, ::-1, :]
        data.append(image_lflip)
        labels.append(label)
        image_vflip = image[::-1, :, :]
        data.append(image_vflip)
        labels.append(label)

        valid_imagepath.append(imagepath)

    indexs = random.sample(list(range(len(valid_imagepath))), 10)
    for index in indexs:
        print('{}\t{}'.format(labels[index*3], valid_imagepath[index]))

    data = np.array(data, dtype = "float") / 255.0
    labels = np.array(labels, dtype = "int")
    labels = tf.keras.utils.to_categorical(labels, num_classes = classes_num)

    trainX, trainY = data, labels
    #(trainX, testX, trainY, testY) = sklearn.model_selection.train_test_split(data, labels, test_size=0.2, random_state=42)

    #trainY = sklearn.preprocessing.LabelBinarizer.fit_transform(trainY)
    #testY  = sklearn.preprocessing.LabelBinarizer.fit_transform(testY)

    return trainX, trainY#, testX, testY

#from keras.utils import np_utils, conv_utils
#from keras.models import Model
#from keras.layers import Flatten, Dense, Input
#from keras.optimizers import Adam
#from keras.applications.resnet50 import ResNet50
#from keras import backend as K

def trans_train():
    save_path = os.path.join(CUR_PATH, 'models/v4_translearning_trainable_weights')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tensorboard_log = os.path.join(save_path, "log")
    if not os.path.exists(tensorboard_log):
        os.makedirs(tensorboard_log)
    image_shape = (image_size, image_size, 3)
    image_batch = 16
    total_epoch = 100
    base_model = keras.applications.resnet50.ResNet50(input_shape=image_shape, weights='imagenet', include_top=False, pooling='max')
    resnet_feature = base_model.output
    fc_softmax = keras.layers.Dense(units=classes_num, activation='softmax')

    model = keras.models.Sequential([
        base_model,
        fc_softmax
        ])
    print(model.summary())
    base_model.trainable = True
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    filepath = os.path.join(save_path, "weights-improvement-{epoch:02d}-{val_acc:.2f}.h5")
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    def scheduler(epoch):
        if epoch % 10 == 0 and epoch != 0:
            lr = keras.backend.get_value(model.optimizer.lr)
            keras.backend.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return keras.backend.get_value(model.optimizer.lr)

    learning_rate = keras.callbacks.LearningRateScheduler(schedule = scheduler, verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, mode='auto', min_lr=base_learning_rate*0.001)
    tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_log, batch_size=image_batch)
    callbacks_list = [checkpoint, reduce_lr, tensorboard]

    #trainX, trainY, testX, testY = get_data(DATA_PATH)
    trainX, trainY = get_data(DATA_PATH)
    print("trainX shape :", trainX.shape)
    print("trainY shape :", trainY.shape)

    training = model.fit(x = trainX, y = trainY,
                    batch_size=image_batch,
                    epochs=total_epoch,
                    verbose = 2,
                    validation_split = 0.2,
                    shuffle = True,
                    callbacks=callbacks_list
                    )

    model.save(os.path.join(save_path, "resnet50_vehicle_watch_classify.h5"))

def train():
    save_path = os.path.join(CUR_PATH, 'models/v3_translearning')
    if os.path.exists(save_path):
        os.makedirs(save_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'
    #input_tensor = keras.layers.Input(shape=(image_size, image_size, 3))
    image_shape = (image_size, image_size, 3)
    image_batch = 16
    total_epoch = 100
    #base_model = keras.applications.resnet50.ResNet50(input_shape=image_shape, include_top=False, weights='imagenet')
    base_model = keras.applications.resnet50.ResNet50(input_shape=image_shape, weights='imagenet', classes=classes_num)
    #base_model.trainable = False

    #resnet_output = base_model.output
    #print(resnet_output.shape)
    #prediction = tf.keras.layers.Dense(1)
    #model = tf.keras.Sequential([
    #    base_model,
    #    prediction_layer
    #])

    #model = keras.Sequential(base_model, keras.layers.Softmax())
    print(base_model.summary())
    keras.utils.plot_model(base_model, to_file = 'resnet50_with-shape.png', show_shapes = True)
    model = base_model
    with open('resnet50_graph_parameters.json', 'w') as f:
        model_json = model.to_json()
        f.write(model_json)
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
#                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    #model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    filepath = os.path.join(save_path, "weights-improvement-{epoch:02d}-{val_acc:.2f}.h5")
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    callbacks_list = [checkpoint]

    #trainX, trainY, testX, testY = get_data(DATA_PATH)
    trainX, trainY = get_data(DATA_PATH)
    print("trainX shape :", trainX.shape)
    print("trainY shape :", trainY.shape)

    #training = model.fit(trainX, trainY, epochs = 100, batch_size = image_batch)

    #initial_epochs = 10
    #validation_steps=20
    #loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)
    #print("initial loss: {:.2f}".format(loss0))
    #print("initial accuracy: {:.2f}".format(accuracy0))

    training = model.fit(x = trainX, y = trainY,
                    batch_size=image_batch,
                    epochs=total_epoch,
                    verbose = 2,
                    validation_split = 0.2,
                    shuffle = True,
                    callbacks=callbacks_list
                    )

    #model.evaluate(testX, testY, batch_size = image_batch * 2)

    model.save(os.path.join(save_path, "resnet50_vehicle_watch_classify.h5"))

def get_image(imageurl):
    #if imagepath.startswith("http"):
    img = skimage.io.imread(imageurl)
    if img is None:
        return None
    dims = len(img.shape)
    if dims == 2:
        img = skimage.color.gray2rgb(img)
    elif dims == 3:
        channel = img.shape[-1]
        if channel == 4:
            img = skimage.color.rgba2rgb(img)
        elif not channel == 3:
            print("invalid image format![{}]".format(img.shape))
            return None
    img = skimage.transform.resize(img, (image_size, image_size))
    img = np.expand_dims(img, axis=0)
    return img

    #model_name = "resnet50_vehicle_watch_classify.h5"
    #model_graph = "resnet50_graph_parameters.json"
    ##model = keras.models.model_from_json(open(model_graph, 'r').read())
    ##model = keras.applications.resnet50.ResNet50(input_shape=(image_size, image_size, 3), weights=None, classes=2)
    #model = keras.models.load_model(model_name)

    ##classes = ['background', 'vehicle', 'watch']
    ##prediction = model.predict(img)
    ##index = np.argmax(prediction)
    ##result = '\t'.join([imageurl, '{}:{:2f}'.format(classes[index], prediction[0][index])])
    ##print(result)
    ##return result
    #keras.preprocessing.image.


def batch_predict(model_name):
    #model_name = "v2/resnet50_vehicle_watch_classify.h5"
    #model_graph = "v2/resnet50_graph_parameters.json"
    #model = keras.models.model_from_json(open(model_graph, 'r').read())
    #model = keras.applications.resnet50.ResNet50(input_shape=(image_size, image_size, 3), weights=None, classes=2)
    model = keras.models.load_model(model_name)

    filename = "haudit_201911_imgurls"
    fileout = os.path.join(CUR_PATH, os.path.dirname(model_name), filename + ".classify_result")
    fout = open(fileout, 'w')
    for line in open(filename, 'r').readlines():
        line = line.strip()
        try:
            img = get_image(line)
            if img is None:
                continue
            classes = ['background', 'vehicle', 'watch']
            prediction = model.predict(img)
            index = np.argmax(prediction)
            if index == 0:
                continue
            result = '\t'.join([line, '{}:{:2f}'.format(classes[index], prediction[0][index])])
            print(result)
            fout.write('%s\n'%result)
        except Exception as e:
            print(e)
            pass

def resume_train(checkpoint_path):
    save_path = os.path.join(CUR_PATH, 'models/v3_translearning')
    checkpoint_path = save_path
    if not os.path.exists(checkpoint_path):
        print("no checkpoint found!")
        return None
    filelist = os.listdir(checkpoint_path)
    filelist.sort(key=lambda fn:os.path.getmtime(checkpoint_path + '/' + fn)
            if not os.path.isdir(os.path.join(checkpoint_path, fn)) else 0)
    latested_checkpoint = os.path.join(checkpoint_path, filelist[-1])
    print("checkpoint exists, load weights from [%s]\n" % latested_checkpoint)

    latested_epoch = int(latested_checkpoint.split('/')[-1].split('-')[2])
    base_learning_rate = 0.0001

    image_shape = (image_size, image_size, 3)
    image_batch = 16
    total_epoch = 100
    base_model = keras.applications.resnet50.ResNet50(input_shape=image_shape, weights=None, include_top=False, pooling='max')

    model = keras.models.Sequential([
        base_model,
        keras.layers.Dense(units=classes_num, activation='softmax')
        ])
    print(model.summary())
    model.load_weights(latested_checkpoint, by_name=True)
    base_model.trainable = False

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    #trainX, trainY, testX, testY = get_data(DATA_PATH)
    trainX, trainY = get_data(DATA_PATH)
    print("trainX shape :", trainX.shape)
    print("trainY shape :", trainY.shape)

    filepath = os.path.join(save_path, "weights-improvement-{epoch:02d}-{val_acc:.2f}.h5")
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False,mode='max')
    callbacks_list = [checkpoint]

    training = model.fit(x = trainX, y = trainY,
                    batch_size=image_batch,
                    epochs=total_epoch,
                    initial_epoch = latested_epoch,
                    verbose = 2,
                    validation_split = 0.2,
                    shuffle = True,
                    callbacks=callbacks_list

                    )

    model.save(os.path.join(save_path, "resnet50_vehicle_watch_classify.h5"))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '3, 2, 1'
    #resume_train('./')
    #train()
    trans_train()
    #batch_predict()
    #get_data(DATA_PATH)






