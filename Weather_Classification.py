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
import matplotlib.pyplot as plt

TRAIN_DIR = 'Train'
TEST_DIR = 'NNTests'
IMG_SIZE = 50
LR = 0.001
MODEL_NAME = 'weather_classification_cnn'


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
    np.save('train_data.npy', training_data)
    return training_data


if os.path.exists('train_data.npy'):  # If you have already created the dataset:
    train_data = np.load('train_data.npy', allow_pickle=True)
else:  # If dataset is not created:
    train_data = create_train_data()

train = train_data

X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
#X_train = X_train/25

y_train = [i[1] for i in train]
Y_train_reshaped = np.array([[int(j == i) for j in range(max(y_train) + 1)] for i in y_train])

tf.reset_default_graph()
conv_input = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

conv1 = conv_2d(conv_input, 32, 5, activation='relu')
pool1 = max_pool_2d(conv1, 5)

conv2 = conv_2d(pool1, 64, 5, activation='relu')
pool2 = max_pool_2d(conv2, 5)

conv3 = conv_2d(pool2, 128, 5, activation='relu')
pool3 = max_pool_2d(conv3, 5)

conv4 = conv_2d(pool3, 64, 5, activation='relu')
pool4 = max_pool_2d(conv4, 5)

conv5 = conv_2d(pool4, 32, 5, activation='relu')
pool5 = max_pool_2d(conv5, 5)


fully_layer = fully_connected(pool5, 1024, activation='relu')
fully_layer = dropout(fully_layer, 0.5)

cnn_layers = fully_connected(fully_layer, 11, activation='softmax')

cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': Y_train_reshaped}, n_epoch=50, snapshot_step=500, show_metric=True,
              run_id=MODEL_NAME)
    model.save('model.tfl')

# img = cv2.imread('Test/Test_1.jpg',0)
# test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
# test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 1)
# prediction = model.predict([test_img])[0]
# for i in range(11):
#     if prediction[i] == max(prediction):
#         print(i)
# print(f"dew: {prediction[0]}, fogsmog: {prediction[1]}, frost: {prediction[2]}, glaze: {prediction[3]}, hail: {prediction[4]}, lightning: {prediction[5]}, "
#       f"rain: {prediction[6]}, rainbow: {prediction[7]}, rime: {prediction[8]}, sandstrom: {prediction[9]}, snow: {prediction[10]}")

test_file = pd.DataFrame(columns=['image_name', 'label'])
index = 0
for test_image in tqdm(sorted(os.listdir(TEST_DIR))):
    test_file.at[index, 'image_name'] = test_image
    path = os.path.join(TEST_DIR, test_image)
    img_test = cv2.imread(path, 0)
    try:
        img_test = cv2.resize(img_test, (IMG_SIZE, IMG_SIZE))
        img_test = img_test.reshape(IMG_SIZE, IMG_SIZE, 1)
        #X_test = X_test / 255
    except:
        continue
    prediction = model.predict([img_test])[0]
    for i in range(11):
        if prediction[i] == max(prediction):
            test_file.at[index, 'label'] = i
    index = index+1
test_file.to_csv('CNN_Predictions_lab.csv', index=False)

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111)
# ax.imshow(test_file, cmap='gray')
# plt.show()
