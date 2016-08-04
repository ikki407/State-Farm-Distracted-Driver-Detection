# -*- coding: utf-8 -*-

import numpy as np

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import random
import h5py
import gc
import time

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold, StratifiedKFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
                                       ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.regularizers import l2, activity_l2

from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
# from keras.layers.normalization import BatchNormalization
# from keras.optimizers import Adam
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
# from sklearn.metrics import log_loss
from numpy.random import permutation

# model name
model_name = 'Model5_VGG19'

# Global setting
np.random.seed(2016)
#use_cache = 1

# color type: 1 - grey, 3 - rgb
color_type_global = 3


# random state for KFold
random_state = 123

# # of CV-fold (= # of training with all data)
nfolds = 15


### For training with data splited by drivers ##


# resize image shape
#img_rows, img_cols = 224, 224
img_rows, img_cols = 224, 224

# batch size and # of epoch
batch_size = 32

nb_epoch_all = 4

nb_epoch = nb_epoch_all
nb_epoch_top_model = nb_epoch_all
# path to the model weights file.
weights_path = '../input/vgg19_weights.h5'
top_model_weights_path = 'fc_model.h5'

### For training with all data
# load mean file
mean = np.load('../input/mean.npy').astype(np.float32)
mean = cv2.resize(mean, (img_rows, img_cols))
#mean = mean.transpose(2,0,1)

split_data = 0.2


datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=13.,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=(0.85, 1.1),
    channel_shift_range=20.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    dim_ordering='th')


# gamma trans.
gamma_trans = True
gamma_rate = (0.75, 1.25)


def do_gamma_trans(img):
    # ガンマ定数の定義
    gamma = random.uniform(gamma_rate[0], gamma_rate[1])
    #print gamma
    look_up_table = np.ones((256, 1), dtype = 'uint8' ) * 0

    for i in range(256):
        look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

    img = cv2.LUT(img, look_up_table)

    return img

# croping image
cropping = False
raw_size = (480, 640)

crop_range = 0.15
crop_size = (int(raw_size[0]*crop_range), int(raw_size[1]*crop_range))

def crop_image(img):
    crop_h = (random.randint(0, crop_size[0]), random.randint(-crop_size[0],-1))
    crop_w = (random.randint(0, crop_size[1]), random.randint(-crop_size[1],-1))

    #print crop_h, crop_w
    #print img.shape
    img = img[crop_h[0]:crop_h[1],crop_w[0]:crop_h[1], :]
    #print img.shape

    return img


if os.path.exists(top_model_weights_path):
    os.remove(top_model_weights_path)

def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_im(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size

    # Cropping before resizing image
    if cropping:
        try:
            img2 = crop_image(img)
            resized = cv2.resize(img2, (img_cols, img_rows))

        except:
            resized = cv2.resize(img, (img_cols, img_rows))

    else:
        resized = cv2.resize(img, (img_cols, img_rows))

    # mean_pixel = [103.939, 116.799, 123.68]
    #resized = resized.astype(np.float32, copy=False)

    # for c in range(3):
    #    resized[:, :, c] = resized[:, :, c] - mean_pixel[c]
    # resized = resized.transpose((2, 0, 1))
    # resized = np.expand_dims(img, axis=0)
    return resized


def get_im_rotate(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    elif color_type == 3:
        img = cv2.imread(path)

    #randomly roated
    #print img
    rotate = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
    #print M
    img = cv2.warpAffine( np.float32(img), np.float32(M),  tuple(np.array([img.shape[1], img.shape[0]])),1)
    #print img
    resized = cv2.resize(np.float32(img), tuple(np.array((img_cols, img_rows), dtype=np.float32)), 1)
    #print resized

    # Reduce size
    #resized = cv2.resize(img, (img_cols, img_rows))
    # mean_pixel = [103.939, 116.799, 123.68]
    #resized = resized.astype(np.float32, copy=False)

    # for c in range(3):
    #    resized[:, :, c] = resized[:, :, c] - mean_pixel[c]
    # resized = resized.transpose((2, 0, 1))
    # resized = np.expand_dims(img, axis=0)
    return resized

def process_line(img_list):
    target = int(img_list[1][1:])
    img_path = '../input/imgs/train/' + img_list[1] + '/' + img_list[2]
    img = get_im(img_path, img_rows, img_cols, color_type=color_type_global)
    return  img, target

def process_line_rotate(img_list):
    target = int(img_list[1][1:])
    img_path = '../input/imgs/train/' + img_list[1] + '/' + img_list[2]
    img = get_im_rotate(img_path, img_rows, img_cols, color_type=color_type_global)
    return  img, target

def image_augmentation(X_train, Y_train, batch_size, datagen=datagen):


    datagen.fit(X_train)

    # fits the model on batches with real-time data augmentation:

    return datagen.flow(X_train, Y_train, batch_size=batch_size).next()

'''
generate_arrays_from_fileとtest_predictionの中身を別関数で定義し、簡潔にする。
特にprocess_lineでtrain,testの時の動きを変える。testではtargetをNoneに

ジェネレータの位置がおかしかったの修正
generate_arrays_from_fileをfor文の中で処理するように変更
test_predictionは変更してない

trainだけを全部読みこんで、image generator でモデルにflowで流す方が速いかも
'''

def generate_arrays_from_file(path, drivers_list=None, \
                        color_type=color_type_global, isvalidation=False, isfinetuning=False,\
                        finetuning_name=None, usingalldata=True):
    while 1:

        #print 'epoch'
        f = open(path)
        f.next() #columns
        #print f
        f_ = list(f)
        f.close()
        f = f_
        if isvalidation==False:
            random.shuffle(f)

        if isfinetuning == True:
            target_id = []

        batch_index = 0
        for line in f:
            if batch_index == 0:
                X_train = []
                y_train = []
            line = line.replace('\n', '').split(',')
            if usingalldata == True:
                if line[0] not in drivers_list:
                    continue
            # create numpy arrays of input data
            # and labels, from each line in the file
            #print line
            if isvalidation == False:
                x, y = process_line(line)

                # gamma trans
                if gamma_trans == True:
                    x = do_gamma_trans(x)

            else:
                x, y = process_line(line)

            if isfinetuning == True:
                target_id.append(y)

            X_train.append(x)
            y_train.append(y)
            batch_index += 1

            if batch_index % batch_size == 0:
                X_train = np.array(X_train, dtype=np.uint8)
                y_train = np.array(y_train, dtype=np.uint8)

                if color_type == 1:
                    X_train = X_train.reshape(X_train.shape[0], color_type,
                                                        img_rows, img_cols)
                else:
                    X_train = X_train.transpose((0, 3, 1, 2))

                y_train = np_utils.to_categorical(y_train, 10)
                X_train = X_train.astype('float32')

                if color_type == 1:
                    X_train /= 255

                else:
                    #X_train /= 255
                    mean_pixel = [103.939, 116.779, 123.68]
                    for c in range(3):
                        X_train[:, c, :, :] = X_train[:, c, :, :] - mean_pixel[c]

                if isvalidation == False:
                    X_train, y_train = image_augmentation(X_train, y_train, batch_index)

                #init
                batch_index = 0
                #print X_train
                yield (X_train, y_train)

        else:

            X_train = np.array(X_train, dtype=np.uint8)
            y_train = np.array(y_train, dtype=np.uint8)

            if color_type == 1:
                X_train = X_train.reshape(X_train.shape[0], color_type,
                                                            img_rows, img_cols)
            else:
                X_train = X_train.transpose((0, 3, 1, 2))

            y_train = np_utils.to_categorical(y_train, 10)
            X_train = X_train.astype('float32')

            if color_type == 1:
                X_train /= 255

            else:
                #X_train /= 255
                mean_pixel = [103.939, 116.779, 123.68]
                for c in range(3):
                    X_train[:, c, :, :] = X_train[:, c, :, :] - mean_pixel[c]

            if isvalidation == False:
                X_train, y_train = image_augmentation(X_train, y_train, batch_index)

            #init
            batch_index = 0

            if isfinetuning == True:
                target_id = np_utils.to_categorical(target_id, 10)
                np.save(open('target_{}.npy'.format(finetuning_name), 'w'), target_id)

            yield (X_train, y_train)

        # close file and shuffle data
        #f.close()


def test_prediction(data_path, color_type=color_type_global, batch_size = 64):

    """
    test_data_generator = test_prediction('../input/imgs/test/*.jpg')
    """
    print('Read test images')

    while 1:

        path = os.path.join(data_path)
        f = glob.glob(path)
        #for debug
        #f = f[:6000]
        X_test = []

        batch_index = 0
        for file_ in f:
            #X_test_id.append(os.path.basename(file_))

            if batch_index == 0:
                X_test = []
            #print line
            x = get_im(file_, img_rows, img_cols, color_type)

            X_test.append(x)
            batch_index += 1

            if batch_index % batch_size == 0:
                X_test = np.array(X_test, dtype=np.uint8)

                if color_type == 1:
                    X_test = X_test.reshape(X_test.shape[0], color_type,
                                                        img_rows, img_cols)
                else:
                    X_test = X_test.transpose((0, 3, 1, 2))

                X_test = X_test.astype('float32')

                if color_type == 1:
                    X_test /= 255

                else:
                    #X_test /= 255
                    mean_pixel = [103.939, 116.779, 123.68]
                    for c in range(3):
                        X_test[:, c, :, :] = X_test[:, c, :, :] - mean_pixel[c]

                #init
                batch_index = 0

                yield X_test

        else:
            X_test = np.array(X_test, dtype=np.uint8)

            if color_type == 1:
                X_test = X_test.reshape(X_test.shape[0], color_type,
                                                    img_rows, img_cols)
            else:
                X_test = X_test.transpose((0, 3, 1, 2))

            X_test = X_test.astype('float32')

            if color_type == 1:
                X_test /= 255

            else:
                #X_test /= 255
                mean_pixel = [103.939, 116.779, 123.68]
                for c in range(3):
                    X_test[:, c, :, :] = X_test[:, c, :, :] - mean_pixel[c]

            #init
            batch_index = 0

            yield X_test

def save_pred(preds, data_path, submission_name='submission'):
    print('Read test images name for submission file')
    path = os.path.join(data_path)
    f = glob.glob(path)
    X_test_id = []

    for file_ in f:
        X_test_id.append(os.path.basename(file_))

    preds_df = pd.DataFrame(preds, columns=['c0', 'c1', 'c2', 'c3',
                                 'c4', 'c5', 'c6', 'c7',
                                            'c8', 'c9'])
    preds_df['img'] = X_test_id

    print 'Saving predictions'
    preds_df.to_csv('submission/' + submission_name + '.csv', index=False)
    return



def averaging(pred_list, submission_name=''):

    cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    pred_ens = pd.DataFrame(np.zeros(79726*10).reshape(79726,10), columns=cols)
    for i in pred_list:
        a = pd.read_csv(i)
        pred_ens[cols] += a[cols]

    pred_ens = pred_ens / len(pred_list)
    pred_ens['img'] = a['img'].values
    pred_ens.to_csv('submission/' + submission_name + '.csv', index=False)




def vgg_std16_model(img_rows, img_cols, color_type=1):

    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape=(color_type, img_rows, img_cols)))
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal',))
                            #input_shape=(color_type, img_rows, img_cols)))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))

    #model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(196, 3, 3, border_mode='same', init='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))

    model.add(Flatten())
    model.add(Dense(128, init='he_normal'))
    model.add(LeakyReLU(alpha=0.001))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, init='he_normal'))
    model.add(LeakyReLU(alpha=0.001))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(Adam(lr=1e-5), loss='categorical_crossentropy')
    return model






def run_simple_cv():

    # get unique drivers
    drivers = pd.read_csv('../input/driver_imgs_list.csv')
    unique_drivers = np.array(list((set(drivers['subject']))))

    kf = KFold(len(unique_drivers), n_folds=nfolds,
                shuffle=True, random_state=random_state)

    num_fold = 0
    fold_number = 0
    cv_pred_list = []
    cv_score = []
    for train_drivers, test_drivers in kf:
        if fold_number == 5:
            fold_number += 1
            continue

        train_drivers_fold = unique_drivers[train_drivers]
        test_drivers_fold = unique_drivers[test_drivers]

        print 'train drivers: {}'.format(train_drivers_fold)
        print 'validation drivers: {}'.format(test_drivers_fold)

        samples_per_epoch = drivers['subject'].isin(train_drivers_fold).sum()
        nb_val_samples = drivers['subject'].isin(test_drivers_fold).sum()

        print 'training data: {}'.format(samples_per_epoch)
        print 'validation data: {}'.format(nb_val_samples)
        #samples_per_epoch = batch_size * (samples_per_epoch // batch_size)
        #nb_val_samples = batch_size * (nb_val_samples // batch_size)

        #
        model = vgg_std16_model(img_rows, img_cols, color_type_global)

        train_data_generator = generate_arrays_from_file( \
                                '../input/driver_imgs_list.csv',
                                train_drivers_fold, isvalidation=False)

        valid_data_generator = generate_arrays_from_file( \
                                '../input/driver_imgs_list.csv',
                                test_drivers_fold, isvalidation=True)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=4, verbose=0),
        ]

        #trianing
        model.fit_generator(train_data_generator,
                    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                    nb_val_samples=nb_val_samples,
                    validation_data=valid_data_generator, max_q_size=10)
                    #callbacks=callbacks)


        predictions_valid = model.evaluate_generator(valid_data_generator,
            val_samples=nb_val_samples)
        #score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', predictions_valid)
        cv_score.append(predictions_valid)

        info_string = 'loss_' + str(predictions_valid) \
                    + '_r_' + str(img_rows) \
                    + '_c_' + str(img_cols) \
                    + '_folds_' + str(fold_number) \
                    + '_ep_' + str(nb_epoch)

        cv_pred_list.append('submission/' + info_string + '.csv')

        # predictions with new version
        test_data_generator = test_prediction('../input/imgs/test/*.jpg')

        preds = model.predict_generator(test_data_generator, val_samples=79726)

        save_pred(preds, '../input/imgs/test/*.jpg', \
                                    submission_name=info_string)

        # next fold
        fold_number += 1


    print 'CV mean: {}, std: {}'.format(np.mean(cv_score), np.std(cv_score))
    averaging(cv_pred_list, 'ensemble_{}'.format(model_name))

    """
    # training with all data
    train_drivers_fold = unique_drivers

    print 'all train drivers: {}'.format(train_drivers_fold)

    samples_per_epoch = drivers['subject'].isin(train_drivers_fold).sum()

    print 'all training data: {}'.format(samples_per_epoch)
    #samples_per_epoch = batch_size * (samples_per_epoch // batch_size)
    #nb_val_samples = batch_size * (nb_val_samples // batch_size)

    #
    model = vgg_std16_model(img_rows, img_cols, color_type_global)

    train_data_generator = generate_arrays_from_file( \
                            '../input/driver_imgs_list.csv',
                            train_drivers_fold, isvalidation=False)

    model.fit_generator(train_data_generator,
                samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,)
                #callbacks=callbacks)

    # predictions with new version
    test_data_generator = test_prediction('../input/imgs/test/*.jpg')

    preds = model.predict_generator(test_data_generator, val_samples=79726)

    save_pred(preds, '../input/imgs/test/*.jpg', \
                                submission_name='all_training_0608.csv')
    """


    return


def save_bottlebeck_features(train_generator, nb_train_samples, \
                                    validation_generator, nb_validation_samples):
    #datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_rows, img_cols)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File('../input/vgg16_weights.h5')
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

    bottleneck_features_validation = model.predict_generator(validation_generator, nb_validation_samples)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)

    return


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.load(open('target_train.npy'))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.load(open('target_valid.npy'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(128, W_regularizer=l2(0.005)))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dropout(0.5))
    model.add(Dense(128, W_regularizer=l2(0.005)))
    model.add(LeakyReLU(alpha=0.001))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy')

    model.fit(train_data, train_labels,
              nb_epoch=nb_epoch_top_model, batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    return


def final_training():
    # build the VGG19 network
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    #model.add(Flatten())
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))


    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')


    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()

    #top_model.add(ZeroPadding2D((1, 1), input_shape=model.output_shape[1:]))
    #top_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv6_1'))
    #top_model.add(ZeroPadding2D((1, 1)))
    #top_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv6_2'))
    #top_model.add(ZeroPadding2D((1, 1)))
    #top_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv6_3'))
    #top_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    top_model.add(Flatten(input_shape=model.output_shape[1:]))

    top_model.add(Dense(2048, init='he_normal'))
    top_model.add(Dropout(0.5))
    top_model.add(BatchNormalization())
    top_model.add(LeakyReLU(alpha=0.0001))


    #top_model.add(Dense(1024, init='he_normal'))
    #top_model.add(Dropout(0.5))
    #top_model.add(BatchNormalization())
    #top_model.add(LeakyReLU(alpha=0.0001))

    top_model.add(Dense(10, init='he_normal'))
    top_model.add(Activation('softmax'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    #top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    #for layer in model.layers[:25]:
    #    layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    #model.compile(loss='categorical_crossentropy',
    #            optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=1e-5))

    #print model.summary()
    return model


def fine_tuning():

    # get unique drivers
    drivers = pd.read_csv('../input/driver_imgs_list.csv')
    unique_drivers = np.array(list((set(drivers['subject']))))

    dlist = list(set(drivers['subject']))
    clist = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

    import itertools
    dc_list = list(itertools.product(dlist, clist))

    random.seed(random_state)
    random.shuffle(dc_list)

    kf = StratifiedKFold(map(lambda x: x[1:],np.array(dc_list)[:,1]), n_folds=nfolds,
                    shuffle=False, random_state=random_state)

    num_fold = 0
    fold_number = 0
    cv_pred_list = []
    cv_score = []

    all_pred_list = []

    for train_drivers, test_drivers in kf:

        #if fold_number <= 0:
        #    fold_number += 1
        #    continue

        '''
        train_drivers_fold = unique_drivers[train_drivers]
        test_drivers_fold = unique_drivers[test_drivers]

        print 'train drivers: {}'.format(train_drivers_fold)
        print 'validation drivers: {}'.format(test_drivers_fold)

        samples_per_epoch = drivers['subject'].isin(train_drivers_fold).sum()
        nb_val_samples = drivers['subject'].isin(test_drivers_fold).sum()

        print 'training data: {}'.format(samples_per_epoch)
        print 'validation data: {}'.format(nb_val_samples)
        #samples_per_epoch = batch_size * (samples_per_epoch // batch_size)
        #nb_val_samples = batch_size * (nb_val_samples // batch_size)

        #
        #model = vgg_std16_model(img_rows, img_cols, color_type_global)

        print 'create generator for training and saving bottlebeck features'
        train_data_generator = generate_arrays_from_file( \
                                '../input/driver_imgs_list.csv',
                                train_drivers_fold, isvalidation=False,\
                                isfinetuning=True, finetuning_name='train')

        valid_data_generator = generate_arrays_from_file( \
                                '../input/driver_imgs_list.csv',
                                test_drivers_fold, isvalidation=True,\
                                isfinetuning=True, finetuning_name='valid')

        print 'save bottlebeck features'
        save_bottlebeck_features(train_data_generator, samples_per_epoch,\
                                      valid_data_generator, nb_val_samples)
        print 'train top model'
        train_top_model()
        '''
        '''
        try:
            print 'Fold{} training with drivers split'.format(fold_number)


            dc_list_tr = np.array(dc_list)[train_drivers].tolist()
            dc_list_te = np.array(dc_list)[test_drivers].tolist()

            print 'combination for validation'
            for i in dc_list_te:
                print i,
            print

            print 'number of validation drivers: {}'.format(len(set(np.array(dc_list_te)[:,0])))
            print 'number of validation class: {}'.format(len(set(np.array(dc_list_te)[:,1])))

            print pd.Series(np.array(dc_list_te)[:,0]).value_counts()
            print pd.Series(np.array(dc_list_te)[:,1]).value_counts()

            def f_tr(data):
                if data.tolist() in dc_list_tr:
                    return True
                else:
                    return False

            def f_te(data):
                if data.tolist() in dc_list_te:
                    return True
                else:
                    return False

            index_tr = drivers[['subject', 'classname']].apply(f_tr, axis=1).values
            index_te = drivers[['subject', 'classname']].apply(f_te, axis=1).values

            alltrain_drivers = drivers[index_tr]
            allvalid_drivers = drivers[index_te]

            # change some validation data to training data
            """allvalid_index = range(len(allvalid_drivers))
            random.seed(407)
            random.shuffle(allvalid_index)

            extract_index = allvalid_index[:len(allvalid_drivers)/2]
            remain_index = allvalid_index[len(allvalid_drivers)/2:]

            alltrain_drivers = pd.concat([alltrain_drivers, allvalid_drivers.iloc[extract_index]], axis=0)
            allvalid_drivers = allvalid_drivers.iloc[remain_index]
            """
            alltrain_drivers.to_csv('../input/driver_imgs_list_alltrain.csv', index=False)
            allvalid_drivers.to_csv('../input/driver_imgs_list_allvalid.csv', index=False)

            print 'final training'
            model = final_training()


            # recalculate

            #print 'train drivers: {}'.format(train_drivers_fold)
            #print 'validation drivers: {}'.format(test_drivers_fold)

            samples_per_epoch = len(alltrain_drivers)
            nb_val_samples = len(allvalid_drivers)

            print 'training data: {}'.format(samples_per_epoch)
            print 'validation data: {}'.format(nb_val_samples)
            #samples_per_epoch = batch_size * (samples_per_epoch // batch_size)
            #nb_val_samples = batch_size * (nb_val_samples // batch_size)

            #
            #model = vgg_std16_model(img_rows, img_cols, color_type_global)

            #print 'create generator for saving bottlebeck features'
            train_data_generator = generate_arrays_from_file( \
                                    '../input/driver_imgs_list_alltrain.csv'
                                    ,isvalidation=False, usingalldata=False)

            valid_data_generator = generate_arrays_from_file( \
                                    '../input/driver_imgs_list_allvalid.csv'
                                    , isvalidation=True, usingalldata=False)

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=4, verbose=0),
            ]

            #trianing
            model.fit_generator(train_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch_all,
                        nb_val_samples=nb_val_samples,
                        validation_data=valid_data_generator, max_q_size=10)


            predictions_valid = model.evaluate_generator(valid_data_generator,
                val_samples=nb_val_samples)
            #score = log_loss(Y_valid, predictions_valid)
            print('Score log_loss: ', predictions_valid)
            cv_score.append(predictions_valid)

            info_string = 'loss_' + str(predictions_valid) \
                        + '_r_' + str(img_rows) \
                        + '_c_' + str(img_cols) \
                        + '_folds_' + str(fold_number) \
                        + '_ep_' + str(nb_epoch)

            cv_pred_list.append('submission/' + info_string + '.csv')

            # predictions with new version
            test_data_generator = test_prediction('../input/imgs/test/*.jpg')

            preds = model.predict_generator(test_data_generator, val_samples=79726)

            save_pred(preds, '../input/imgs/test/*.jpg', \
                                        submission_name=info_string)

            del model
            gc.collect()


        except Exception as e:
            print str(e)



        # delete top model weights
        if os.path.exists(top_model_weights_path):
            os.remove(top_model_weights_path)
        '''

        ### Using all data with random split
        try:
            print 'Fold{} training with all data'.format(fold_number)

            drivers = pd.read_csv('../input/driver_imgs_list.csv')

            # random split
            np.random.seed(fold_number)
            split_data = np.random.uniform(0.05, 0.15)

            random_index = random.sample(range(len(drivers)),int(len(drivers)*(1-split_data)))
            alltrain_drivers = drivers.iloc[(drivers.index.isin(random_index)), :]
            allvalid_drivers = drivers.iloc[~(drivers.index.isin(random_index)), :]

            alltrain_drivers.to_csv('../input/driver_imgs_list_alltrain.csv', index=False)
            allvalid_drivers.to_csv('../input/driver_imgs_list_allvalid.csv', index=False)


            print 'final training'
            model = final_training()


            # recalculate

            #print 'train drivers: {}'.format(train_drivers_fold)
            #print 'validation drivers: {}'.format(test_drivers_fold)

            samples_per_epoch = len(alltrain_drivers)
            nb_val_samples = len(allvalid_drivers)

            print 'training data: {}'.format(samples_per_epoch)
            print 'validation data: {}'.format(nb_val_samples)
            #samples_per_epoch = batch_size * (samples_per_epoch // batch_size)
            #nb_val_samples = batch_size * (nb_val_samples // batch_size)

            #
            #model = vgg_std16_model(img_rows, img_cols, color_type_global)

            #print 'create generator for saving bottlebeck features'
            train_data_generator = generate_arrays_from_file( \
                                    '../input/driver_imgs_list_alltrain.csv'
                                    ,isvalidation=False, usingalldata=False)

            valid_data_generator = generate_arrays_from_file( \
                                    '../input/driver_imgs_list_allvalid.csv'
                                    , isvalidation=True, usingalldata=False)

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=4, verbose=0),
            ]

            #trianing
            model.fit_generator(train_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch_all,
                        nb_val_samples=nb_val_samples,
                        validation_data=valid_data_generator, max_q_size=10)
                        #callbacks=callbacks)


            predictions_valid = model.evaluate_generator(valid_data_generator,
                val_samples=nb_val_samples)
            #score = log_loss(Y_valid, predictions_valid)
            print('Score log_loss: ', predictions_valid)
            #cv_score.append(predictions_valid)

            info_string = 'all_loss_' + str(predictions_valid) \
                        + '_r_' + str(img_rows) \
                        + '_c_' + str(img_cols) \
                        + '_folds_' + str(fold_number) \
                        + '_ep_' + str(nb_epoch_all)

            all_pred_list.append('submission/' + info_string + '.csv')

            # predictions with new version
            test_data_generator = test_prediction('../input/imgs/test/*.jpg')

            preds = model.predict_generator(test_data_generator, val_samples=79726)

            save_pred(preds, '../input/imgs/test/*.jpg', \
                                        submission_name=info_string)

            del model
            gc.collect()


        except Exception as e:
            print str(e)
            fold_number += 1
            continue


        # next fold
        fold_number += 1


    #print 'CV mean: {:.6}, std: {:.6}'.format(np.mean(cv_score), np.std(cv_score))
    #averaging(cv_pred_list, 'ensemble_{}_CV{:.3}'.format(model_name, np.mean(cv_score)))
    averaging(all_pred_list, 'ensemble_{}_all'.format(model_name))

    #cv_all_pred_list = cv_pred_list[:]
    #cv_all_pred_list.extend(all_pred_list)
    #averaging(cv_all_pred_list, 'ensemble_{}_CV_all'.format(model_name))


    return

if __name__ == '__main__':
    # run own defined model
    #run_simple_cv()

    # run fine-tuned model
    fine_tuning()
