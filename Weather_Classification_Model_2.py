import cv2
import pandas as pd
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = 'Train'
TEST_DIR = 'NNTests'
IMG_SIZE = 28
LR = 0.01
MODEL_NAME = 'weather_classification_LENet'

def create_train_data():
    training_data = []
    class_label = 0
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        for img in tqdm(sorted(os.listdir(os.path.join(TRAIN_DIR, folder)))):
            path = os.path.join(TRAIN_DIR, folder, img)
            img_data = cv2.imread(path, 0)
            try:
                img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
            except:
                continue
            training_data.append([np.array(img_data), class_label])
        class_label += 1
    shuffle(training_data)
    np.save('train_data_2.npy', training_data)
    return training_data

if os.path.exists('train_data_2.npy'):  # If you have already created the dataset:
    train_data = np.load('train_data_2.npy', allow_pickle=True)
else:  # If dataset is not created:
    train_data = create_train_data()

train = train_data

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train = [i[1] for i in train]
Y_train_reshaped = np.array([[int(j == i) for j in range(max(y_train) + 1)] for i in y_train])

tf.reset_default_graph()

conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

conv1 = conv_2d(conv_input, 20, 5, strides=1, activation='relu')
pool1 = max_pool_2d(conv1, 2, strides=2)

conv2 = conv_2d(pool1, 20, 5, strides=1, activation='relu')
pool2 = max_pool_2d(conv2, 2, strides=2)

fully_layer1 = fully_connected(pool2, 500, activation='relu')
fully_layer1 = dropout(fully_layer1, 0.5)

fully_layer2 = fully_connected(fully_layer1, 500, activation='relu')
fully_layer2 = dropout(fully_layer2, 0.5)

LeNet_layers = fully_connected(fully_layer2, 11, activation='softmax')

LeNet_layers = regression(LeNet_layers, optimizer='SGD', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model2 = tflearn.DNN(LeNet_layers, tensorboard_dir='log2', tensorboard_verbose=3)

if os.path.exists('model2.tfl.meta'):
    model2.load('./model2.tfl')
else:
    model2.fit({'input': X_train}, {'targets': Y_train_reshaped}, n_epoch=30, snapshot_step=500, show_metric=True,
              run_id=MODEL_NAME)
    model2.save('model2.tfl')

test_file = pd.DataFrame(columns=['image_name', 'label'])
index = 0
for test_image in tqdm(sorted(os.listdir(TEST_DIR))):
    test_file.at[index, 'image_name'] = test_image
    path = os.path.join(TEST_DIR, test_image)
    img_test = cv2.imread(path, 0)
    try:
        img_test = cv2.resize(img_test, (IMG_SIZE, IMG_SIZE))
        img_test = img_test.reshape(IMG_SIZE, IMG_SIZE, 1)
    except:
        continue
    prediction = model2.predict([img_test])[0]
    for i in range(11):
        if prediction[i] == max(prediction):
            test_file.at[index, 'label'] = i
    index = index+1
test_file.to_csv('LeNet_Predictions_Lab2.csv', index=False)
# # conv1 = conv_2d(conv_input, 64, 7, activation='relu')
# # conv2 = conv_2d(conv1, 32, 7, activation='relu')
# # pool2 = max_pool_2d(conv2, 7)
# # conv3 = conv_2d(pool2, 64, 7, activation='relu')
# #
# # inc_conv1 = conv_2d(conv3, 64, 1, activation='relu')
# #
# # inc_conv2 = conv_2d(conv3, 64, 1, activation='relu')
# # inc_conv22 = conv_2d(inc_conv2, 32, 3, activation='relu')
# #
# # inc_conv3 = conv_2d(conv3, 64, 1, activation='relu')
# # inc_conv33 = conv_2d(inc_conv3, 32, 5, activation='relu')
# #
# # pool4 = max_pool_2d(conv3, 5)
# # inc_conv4 = conv_2d(pool4, 64, 1, activation='relu')
# #
# # mixed = concatenate([inc_conv1, inc_conv22, inc_conv33, inc_conv4], axis=-1)
#
# conv1 = conv_2d(conv_input, 64, 3, strides=1, padding='same', activation='relu')
# conv2 = conv_2d(conv1, 64, 3, strides=1, padding='same', activation='relu')
# pool1 = max_pool_2d(conv2, 2, strides=2)
#
# conv3 = conv_2d(pool1, 128, 3, strides=1, padding='same', activation='relu')
# conv4 = conv_2d(conv3, 128, 3, strides=1, padding='same', activation='relu')
# pool2 = max_pool_2d(conv4, 2, strides=2)
#
# conv4 = conv_2d(pool2, 256, 3, strides=1, padding='same', activation='relu')
# conv5 = conv_2d(conv4, 256, 3, strides=1, padding='same', activation='relu')
# conv6 = conv_2d(conv5, 256, 3, strides=1, padding='same', activation='relu')
# pool3 = max_pool_2d(conv6, 2, strides=2)
#
# conv7 = conv_2d(pool3, 512, 3, strides=1, padding='same', activation='relu')
# conv8 = conv_2d(conv7, 512, 3, strides=1, padding='same', activation='relu')
# conv9 = conv_2d(conv8, 512, 3, strides=1, padding='same', activation='relu')
# pool4 = max_pool_2d(conv9, 2, strides=2)
#
# conv10 = conv_2d(pool4, 512, 3, strides=1, padding='same', activation='relu')
# conv11 = conv_2d(conv10, 512, 3, strides=1, padding='same', activation='relu')
# conv12 = conv_2d(conv11, 512, 3, strides=1, padding='same', activation='relu')
# pool5 = max_pool_2d(conv8, 2, strides=2)
#
# fully_layer1 = fully_connected(pool5, 4096, activation='relu')
# fully_layer1 = dropout(fully_layer1, 0.5)
#
# fully_layer2 = fully_connected(fully_layer1, 4096, activation='relu')
# fully_layer2 = dropout(fully_layer2, 0.5)
#
# fully_layer3 = fully_connected(fully_layer2, 1000, activation='softmax')
# VGG_layers = fully_connected(fully_layer3, 11, activation='softmax')
#
# VGG_layers = regression(VGG_layers, optimizer='SGD', learning_rate=LR, loss='categorical_crossentropy', name='targets')
#
# model2 = tflearn.DNN(VGG_layers, tensorboard_dir='log2', tensorboard_verbose=3)
#
# if os.path.exists('model2.tfl.meta'):
#     model2.load('./model2.tfl')
# else:
#     model2.fit({'input': X_train}, {'targets': Y_train_reshaped}, n_epoch=10, snapshot_step=500, show_metric=True,
#               run_id=MODEL_NAME)
#     model2.save('model2.tfl')
#
# test_file = pd.DataFrame(columns=['image_name', 'label'])
# index = 0
# for test_image in tqdm(sorted(os.listdir(TEST_DIR))):
#     test_file.at[index, 'image_name'] = test_image
#     path = os.path.join(TEST_DIR, test_image)
#     img_test = cv2.imread(path, 0)
#     try:
#         img_test = cv2.resize(img_test, (IMG_SIZE, IMG_SIZE))
#         img_test = img_test.reshape(IMG_SIZE, IMG_SIZE, 1)
#     except:
#         continue
#     prediction = model2.predict([img_test])[0]
#     for i in range(11):
#         if prediction[i] == max(prediction):
#             test_file.at[index, 'label'] = i
#     index = index+1
# test_file.to_csv('VGG_Predictions.csv', index=False)
