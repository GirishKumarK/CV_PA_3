import os
import sys
import time
import random
import numpy as np
import pandas as pd
import skvideo.io as skvio
from sklearn.utils import shuffle
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

def get_data_ref():
    # Get Dataset Reference Location
    for dirName, subdirList, fileList in os.walk('.', topdown=True):
        for fname in fileList:
            if (fname == 'action_label.csv'):
                data_src = os.path.join(dirName, fname)
    # Return the Dataset Reference Path
    return data_src

def get_data_info():
    # Get the Information from Dataset Reference File
    try:
        data_src = get_data_ref()
        print ('Dataset Reference File Found In Location : ' + data_src)
        dataset = pd.read_csv(data_src)
    except:
        for p in range(5):
            print ('File Not Found ! Directory Does Not Exist ! Re-make Dataset !')
        print ('~' * 90)
        sys.exit('File Not Found ! Directory Does Not Exist ! Re-make Dataset !')
    return dataset

def get_dataset_splits():
    # A Method to Read the Raw Dataset and Return Dataset Splits
    # Get Dataset
    dataset = get_data_info()
    # Convert to NumPy Array
    dataset = np.array(dataset)
    # Get Action Videos
    videos = dataset[:, 0].tolist()
    # Get Action Labels
    labels = dataset[:, 1].tolist()
    # Get Label Classes
    classes = np.unique(labels).tolist()
    # Separate Out Each Class Videos
    actions = []
    for n in range(len(classes)):
        action = np.where(np.array(labels) == classes[n])[0]
        actions.append(action.tolist())
    # Pick One Video from Every Action Class as Test Dataset
    test_data = []
    for a in actions:
        # Choose a Random Index
        index = random.choice(a)
        # Add the Chosen Index Value of Video and Label to Test Set
        test_data.append([videos[index], labels[index]])
        # Delete the Chosen Values from Actions List
        a.pop(a.index(index))
    # Pick One Video from Every Action Class as Validation Dataset
    valid_data = []
    for a in actions:
        # Choose a Random Index
        index = random.choice(a)
        # Add the Chosen Index Value of Video and Label to Test Set
        valid_data.append([videos[index], labels[index]])
        # Delete the Chosen Values from Actions List
        a.pop(a.index(index))
    # Merge The Action Label Pairs as Train Dataset
    train_data = []
    for a in actions:
        for v in a:
            # Generate Action Label Pair and Add to Train Dataset
            train_data.append([videos[v], labels[v]])
    # Shuffle Train Data To Randomize the Ordered Labels
    train_data = shuffle(train_data)
    # Return the DataFrame Splits of Train Validation Test
    train_data = pd.DataFrame(train_data, columns=['Action', 'Label'])
    valid_data = pd.DataFrame(valid_data, columns=['Action', 'Label'])
    test_data = pd.DataFrame(test_data, columns=['Action', 'Label'])
    return train_data, valid_data, test_data

def get_shuffled_data(dataset):
    # Shuffle the Dataset to Randomize the Ordered Labels
    shuffled_data = shuffle(dataset)
    return shuffled_data

def get_video_specs(video_file_name):
    # Search and Obtain Video from Video File Name
    vids_dir = get_data_ref().split('\\')
    vids_dir = os.path.join(vids_dir[0], vids_dir[1])
    video_file = os.path.join(vids_dir, video_file_name)
    # Normalizing All Video Frame Sizes to h360 x w480
    video = skvio.vread(video_file, as_grey=True, outputdict={"-sws_flags": "bilinear", "-s": "480x360"}) # -s : width x height
    frames, height, width, channels = video.shape
    return video, frames, height, width, channels

def get_videos_as_images_with_labels(data_split, split_name):
    # Obtain the Videos as Frames along with Labels
    # Get Directory of Action Vidos
    vids_dir = get_data_ref().split('\\')
    vids_dir = os.path.join(vids_dir[0], vids_dir[1])
    # Get the List of Video Names
    videos = np.array([[row['Action']] for index, row in data_split.iterrows()])
    # Get the List of Labels of Every Video
    labels = np.array([[row['Label']] for index, row in data_split.iterrows()]).ravel()
    # Encode the List of Labels to Numeric Value
    le = preprocessing.LabelEncoder()
    le.fit(labels) # lfit return
    labels_encoded = le.transform(labels)
    # Split Video into Frames and Get Video Specs
    tss = time.time()
    frame_spec_label = []
    for vdo in range(len(videos)):
        images, frames, height, width, channels = get_video_specs(videos[vdo][0])
        for img in images:
            #image = img.ravel()
            image = img.reshape(1, height, width, channels)
            frame_spec_label.append([image, height, width, channels, labels_encoded[vdo]])
    frame_spec_label = np.array(frame_spec_label)
    tes = time.time()
    print ('Time Taken To Get %s Frame_Specs_Label List : %f Mins.' % (split_name, ((tes - tss) / 60.0)))
    return frame_spec_label

def set_placeholders():
    # Setting of Placeholders for Input Videos
    num_classes = 13
    ph_action = tf.placeholder(tf.float32, [None, 360, 480, 1])
    ph_label = tf.placeholder(tf.float32, [None, num_classes])
    ph_train = tf.placeholder(tf.bool)
    # Return Placeholders
    return ph_action, ph_label, ph_train

def my_cnn(ph_action, ph_train):
    # My Convolutional Neural Network Model
    # Set Constants
    decay_rate = 0.01
    #drop_rate = 0.50
    dense_units = 1024
    # Regularizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=decay_rate)
    # Network Architecture
    # Incoming Image Flattening
    flat0 = tf.reshape(ph_action, [-1, 360, 480, 1])
    # First Convolution Branch
    conv1a = tf.layers.conv2d(inputs=flat0, filters=12, kernel_size=[18, 24], padding='valid', activation=tf.nn.relu)
    conv1b = tf.layers.conv2d(inputs=flat0, filters=12, kernel_size=[18, 24], padding='valid', activation=tf.nn.relu)
    conv1c = tf.layers.conv2d(inputs=flat0, filters=12, kernel_size=[18, 24], padding='valid', activation=tf.nn.relu)
    conv1d = tf.layers.conv2d(inputs=flat0, filters=12, kernel_size=[18, 24], padding='valid', activation=tf.nn.relu)
    # First Maximum Pooling Branch
    pool1a = tf.layers.max_pooling2d(inputs=conv1a, pool_size=[2, 2], strides=4)
    pool1b = tf.layers.max_pooling2d(inputs=conv1b, pool_size=[2, 2], strides=4)
    pool1c = tf.layers.max_pooling2d(inputs=conv1c, pool_size=[2, 2], strides=4)
    pool1d = tf.layers.max_pooling2d(inputs=conv1d, pool_size=[2, 2], strides=4)
    # Second Convolution Branch
    conv2a1 = tf.layers.conv2d(inputs=pool1a, filters=16, kernel_size=[8, 11], padding='valid', activation=tf.nn.relu)
    conv2a2 = tf.layers.conv2d(inputs=pool1a, filters=16, kernel_size=[8, 11], padding='valid', activation=tf.nn.relu)
    conv2b1 = tf.layers.conv2d(inputs=pool1b, filters=16, kernel_size=[8, 11], padding='valid', activation=tf.nn.relu)
    conv2b2 = tf.layers.conv2d(inputs=pool1b, filters=16, kernel_size=[8, 11], padding='valid', activation=tf.nn.relu)
    conv2c1 = tf.layers.conv2d(inputs=pool1c, filters=16, kernel_size=[8, 11], padding='valid', activation=tf.nn.relu)
    conv2c2 = tf.layers.conv2d(inputs=pool1c, filters=16, kernel_size=[8, 11], padding='valid', activation=tf.nn.relu)
    conv2d1 = tf.layers.conv2d(inputs=pool1d, filters=16, kernel_size=[8, 11], padding='valid', activation=tf.nn.relu)
    conv2d2 = tf.layers.conv2d(inputs=pool1d, filters=16, kernel_size=[8, 11], padding='valid', activation=tf.nn.relu)
    # Second Maximum Pooling Branch
    pool2a1 = tf.layers.max_pooling2d(inputs=conv2a1, pool_size=[2, 2], strides=3)
    pool2a2 = tf.layers.max_pooling2d(inputs=conv2a2, pool_size=[2, 2], strides=3)
    pool2b1 = tf.layers.max_pooling2d(inputs=conv2b1, pool_size=[2, 2], strides=3)
    pool2b2 = tf.layers.max_pooling2d(inputs=conv2b2, pool_size=[2, 2], strides=3)
    pool2c1 = tf.layers.max_pooling2d(inputs=conv2c1, pool_size=[2, 2], strides=3)
    pool2c2 = tf.layers.max_pooling2d(inputs=conv2c2, pool_size=[2, 2], strides=3)
    pool2d1 = tf.layers.max_pooling2d(inputs=conv2d1, pool_size=[2, 2], strides=3)
    pool2d2 = tf.layers.max_pooling2d(inputs=conv2d2, pool_size=[2, 2], strides=3)
    # Third Convolution Branch
    conv3a1a = tf.layers.conv2d(inputs=pool2a1, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3a1b = tf.layers.conv2d(inputs=pool2a1, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3a2a = tf.layers.conv2d(inputs=pool2a2, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3a2b = tf.layers.conv2d(inputs=pool2a2, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3b1a = tf.layers.conv2d(inputs=pool2b1, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3b1b = tf.layers.conv2d(inputs=pool2b1, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3b2a = tf.layers.conv2d(inputs=pool2b2, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3b2b = tf.layers.conv2d(inputs=pool2b2, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3c1a = tf.layers.conv2d(inputs=pool2c1, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3c1b = tf.layers.conv2d(inputs=pool2c1, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3c2a = tf.layers.conv2d(inputs=pool2c2, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3c2b = tf.layers.conv2d(inputs=pool2c2, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3d1a = tf.layers.conv2d(inputs=pool2d1, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3d1b = tf.layers.conv2d(inputs=pool2d1, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3d2a = tf.layers.conv2d(inputs=pool2d2, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv3d2b = tf.layers.conv2d(inputs=pool2d2, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    # Fourth Convolution Layer
    conv4a1a = tf.layers.conv2d(inputs=conv3a1a, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4a1b = tf.layers.conv2d(inputs=conv3a1b, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4a2a = tf.layers.conv2d(inputs=conv3a2a, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4a2b = tf.layers.conv2d(inputs=conv3a2b, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4b1a = tf.layers.conv2d(inputs=conv3b1a, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4b1b = tf.layers.conv2d(inputs=conv3b1b, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4b2a = tf.layers.conv2d(inputs=conv3b2a, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4b2b = tf.layers.conv2d(inputs=conv3b2b, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4c1a = tf.layers.conv2d(inputs=conv3c1a, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4c1b = tf.layers.conv2d(inputs=conv3c1b, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4c2a = tf.layers.conv2d(inputs=conv3c2a, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4c2b = tf.layers.conv2d(inputs=conv3c2b, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4d1a = tf.layers.conv2d(inputs=conv3d1a, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4d1b = tf.layers.conv2d(inputs=conv3d1b, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4d2a = tf.layers.conv2d(inputs=conv3d2a, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv4d2b = tf.layers.conv2d(inputs=conv3d2b, filters=12, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    # Fifth Convolutional Branch
    conv5a1a = tf.layers.conv2d(inputs=conv4a1a, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5a1b = tf.layers.conv2d(inputs=conv4a1b, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5a2a = tf.layers.conv2d(inputs=conv4a2a, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5a2b = tf.layers.conv2d(inputs=conv4a2b, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5b1a = tf.layers.conv2d(inputs=conv4b1a, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5b1b = tf.layers.conv2d(inputs=conv4b1b, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5b2a = tf.layers.conv2d(inputs=conv4b2a, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5b2b = tf.layers.conv2d(inputs=conv4b2b, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5c1a = tf.layers.conv2d(inputs=conv4c1a, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5c1b = tf.layers.conv2d(inputs=conv4c1b, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5c2a = tf.layers.conv2d(inputs=conv4c2a, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5c2b = tf.layers.conv2d(inputs=conv4c2b, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5d1a = tf.layers.conv2d(inputs=conv4d1a, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5d1b = tf.layers.conv2d(inputs=conv4d1b, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5d2a = tf.layers.conv2d(inputs=conv4d2a, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    conv5d2b = tf.layers.conv2d(inputs=conv4d2b, filters=8, kernel_size=[3, 4], padding='valid', activation=tf.nn.relu)
    # Third Maximum Pooling Branch
    pool3a1a = tf.layers.max_pooling2d(inputs=conv5a1a, pool_size=[2, 2], strides=3)
    pool3a1b = tf.layers.max_pooling2d(inputs=conv5a1b, pool_size=[2, 2], strides=3)
    pool3a2a = tf.layers.max_pooling2d(inputs=conv5a2a, pool_size=[2, 2], strides=3)
    pool3a2b = tf.layers.max_pooling2d(inputs=conv5a2b, pool_size=[2, 2], strides=3)
    pool3b1a = tf.layers.max_pooling2d(inputs=conv5b1a, pool_size=[2, 2], strides=3)
    pool3b1b = tf.layers.max_pooling2d(inputs=conv5b1b, pool_size=[2, 2], strides=3)
    pool3b2a = tf.layers.max_pooling2d(inputs=conv5b2a, pool_size=[2, 2], strides=3)
    pool3b2b = tf.layers.max_pooling2d(inputs=conv5b2b, pool_size=[2, 2], strides=3)
    pool3c1a = tf.layers.max_pooling2d(inputs=conv5c1a, pool_size=[2, 2], strides=3)
    pool3c1b = tf.layers.max_pooling2d(inputs=conv5c1b, pool_size=[2, 2], strides=3)
    pool3c2a = tf.layers.max_pooling2d(inputs=conv5c2a, pool_size=[2, 2], strides=3)
    pool3c2b = tf.layers.max_pooling2d(inputs=conv5c2b, pool_size=[2, 2], strides=3)
    pool3d1a = tf.layers.max_pooling2d(inputs=conv5d1a, pool_size=[2, 2], strides=3)
    pool3d1b = tf.layers.max_pooling2d(inputs=conv5d1b, pool_size=[2, 2], strides=3)
    pool3d2a = tf.layers.max_pooling2d(inputs=conv5d2a, pool_size=[2, 2], strides=3)
    pool3d2b = tf.layers.max_pooling2d(inputs=conv5d2b, pool_size=[2, 2], strides=3)
    # First Flattening Branch
    flat1a1a = tf.reshape(pool3a1a, [-1, int(pool3a1a.shape[1]) * int(pool3a1a.shape[2]) * int(pool3a1a.shape[3])])
    flat1a1b = tf.reshape(pool3a1b, [-1, int(pool3a1b.shape[1]) * int(pool3a1b.shape[2]) * int(pool3a1b.shape[3])])
    flat1a2a = tf.reshape(pool3a2a, [-1, int(pool3a2a.shape[1]) * int(pool3a2a.shape[2]) * int(pool3a2a.shape[3])])
    flat1a2b = tf.reshape(pool3a2b, [-1, int(pool3a2b.shape[1]) * int(pool3a2b.shape[2]) * int(pool3a2b.shape[3])])
    flat1b1a = tf.reshape(pool3b1a, [-1, int(pool3b1a.shape[1]) * int(pool3b1a.shape[2]) * int(pool3b1a.shape[3])])
    flat1b1b = tf.reshape(pool3b1b, [-1, int(pool3b1b.shape[1]) * int(pool3b1b.shape[2]) * int(pool3b1b.shape[3])])
    flat1b2a = tf.reshape(pool3b2a, [-1, int(pool3b2a.shape[1]) * int(pool3b2a.shape[2]) * int(pool3b2a.shape[3])])
    flat1b2b = tf.reshape(pool3b2b, [-1, int(pool3b2b.shape[1]) * int(pool3b2b.shape[2]) * int(pool3b2b.shape[3])])
    flat1c1a = tf.reshape(pool3c1a, [-1, int(pool3c1a.shape[1]) * int(pool3c1a.shape[2]) * int(pool3c1a.shape[3])])
    flat1c1b = tf.reshape(pool3c1b, [-1, int(pool3c1b.shape[1]) * int(pool3c1b.shape[2]) * int(pool3c1b.shape[3])])
    flat1c2a = tf.reshape(pool3c2a, [-1, int(pool3c2a.shape[1]) * int(pool3c2a.shape[2]) * int(pool3c2a.shape[3])])
    flat1c2b = tf.reshape(pool3c2b, [-1, int(pool3c2b.shape[1]) * int(pool3c2b.shape[2]) * int(pool3c2b.shape[3])])
    flat1d1a = tf.reshape(pool3d1a, [-1, int(pool3d1a.shape[1]) * int(pool3d1a.shape[2]) * int(pool3d1a.shape[3])])
    flat1d1b = tf.reshape(pool3d1b, [-1, int(pool3d1b.shape[1]) * int(pool3d1b.shape[2]) * int(pool3d1b.shape[3])])
    flat1d2a = tf.reshape(pool3d2a, [-1, int(pool3d2a.shape[1]) * int(pool3d2a.shape[2]) * int(pool3d2a.shape[3])])
    flat1d2b = tf.reshape(pool3d2b, [-1, int(pool3d2b.shape[1]) * int(pool3d2b.shape[2]) * int(pool3d2b.shape[3])])
    # First Dense Branch
    dense1a1a = tf.layers.dense(inputs=flat1a1a, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1a1b = tf.layers.dense(inputs=flat1a1b, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1a2a = tf.layers.dense(inputs=flat1a2a, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1a2b = tf.layers.dense(inputs=flat1a2b, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1b1a = tf.layers.dense(inputs=flat1b1a, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1b1b = tf.layers.dense(inputs=flat1b1b, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1b2a = tf.layers.dense(inputs=flat1b2a, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1b2b = tf.layers.dense(inputs=flat1b2b, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1c1a = tf.layers.dense(inputs=flat1c1a, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1c1b = tf.layers.dense(inputs=flat1c1b, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1c2a = tf.layers.dense(inputs=flat1c2a, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1c2b = tf.layers.dense(inputs=flat1c2b, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1d1a = tf.layers.dense(inputs=flat1d1a, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1d1b = tf.layers.dense(inputs=flat1d1b, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1d2a = tf.layers.dense(inputs=flat1d2a, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    dense1d2b = tf.layers.dense(inputs=flat1d2b, units=128, activation=tf.nn.relu, kernel_regularizer=regularizer)
    # Second Dense Branch
    dense_list = [dense1a1a, dense1a1b, dense1a2a, dense1a2b, dense1b1a, dense1b1b, dense1b2a, dense1b2b, 
                  dense1c1a, dense1c1b, dense1c2a, dense1c2b, dense1d1a, dense1d1b, dense1d2a, dense1d2b]
    dense1 = tf.concat(values=dense_list, axis=0)
    dense2 = tf.layers.dense(inputs=dense1, units=dense_units, activation=tf.nn.relu, kernel_regularizer=regularizer)
    # Fully Connected Layer
    fully_connected_layer = dense2
    # Return Final Fully Connected Layer
    return fully_connected_layer, dense_units

def get_logits(fully_connected_layer, dense_units):
    # Get The Logits Output
    num_classes = 13
    w = tf.Variable(tf.truncated_normal([dense_units, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    logits = tf.matmul(fully_connected_layer, w) + b
    logits = tf.nn.softmax(logits)
    return logits

def get_loss(ph_label, logits):
    # Get The Loss In The Model
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ph_label, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def get_train_op(loss, learning_rate):
    # Get The Training Optimizer Model
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss)
    return train_op

def get_correct_predictions(ph_label, logits):
    # Get The Correct Predictions For The Model
    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(ph_label, 1))
    correct_predictions = tf.cast(correct_predictions, tf.float32)
    return correct_predictions

def get_accuracy(ph_label, logits):
    # Get The Accuracy For The Model
    correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(ph_label, 1))
    correct_predictions = tf.cast(correct_predictions, tf.float32)
    accuracy = tf.reduce_mean(correct_predictions)
    return accuracy

def get_predictions(logits):
    # Get Predictions Of Dataset
    predictions = tf.argmax(logits, 1)
    return predictions

def set_gpu_props():
    # Set The Properties For GPU TensorFlow
    has_GPU = True # Set To False If No GPU
    if has_GPU:
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, 
                                   allocator_type='BFC', 
                                   deferred_deletion_bytes=0, 
                                   allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_option)
    else:
        config = tf.ConfigProto()
    return config

def init_sess(gpu_config):
    # Create Session for TensorFlow
    session = tf.Session(config=gpu_config)
    # Initialize Global Variables
    session.run(tf.global_variables_initializer())
    # Return Session
    return session

def train_cnn(session, frame_spec_label_train, accuracy, ph_action, ph_label, ph_train):
    # Splitting the Data into Smaller Parts to avoid Memory Overflow in GPU
    # Train using The Train Dataset Split
    frame_spec_label_train = shuffle(frame_spec_label_train)
    splits = int(frame_spec_label_train.shape[0] / 1)
    batch = int(frame_spec_label_train.shape[0] / splits)
    accuracies = []
    tstr = time.time()
    for s in range(splits):
        batch_accuracy = session.run(accuracy, feed_dict={ph_action: frame_spec_label_train[batch*s:(batch*(s+1))+1, 0][0], ph_label: frame_spec_label_train[batch*s:(batch*(s+1))+1, 4][0]*np.ones(13).reshape(1, 13), ph_train: True})
        accuracies.append(batch_accuracy)
    total_train_acc = sum(accuracies) / len(accuracies)
    tetr = time.time()
    print ('Training Accuracy : %f ::: Training Time : %f Secs.' % ((total_train_acc), (tetr - tstr)))
    # Return Train Accuracy
    return total_train_acc

def validate_cnn(session, frame_spec_label_valid, accuracy, ph_action, ph_label, ph_train):
    # Splitting the Data into Smaller Parts to avoid Memory Overflow in GPU
    # Testing the Trained Model on Validation Dataset Split
    frame_spec_label_valid = shuffle(frame_spec_label_valid)
    splits = int(frame_spec_label_valid.shape[0] / 1)
    batch = int(frame_spec_label_valid.shape[0] / splits)
    accuracies = []
    tsv = time.time()
    for j in range(splits):
        batch_accuracy = session.run(accuracy, feed_dict={ph_action: frame_spec_label_valid[batch*j:(batch*(j+1))+1, 0][0], ph_label: frame_spec_label_valid[batch*j:(batch*(j+1))+1, 4][0]*np.ones(13).reshape(1, 13), ph_train: False})
        accuracies.append(batch_accuracy)
    total_valid_acc = sum(accuracies) / len(accuracies)
    tev = time.time()
    print ('Validation Accuracy : %f ::: Validation Time : %f Secs.' % (total_valid_acc, (tev - tsv)))
    # Return Validation Accuracy
    return total_valid_acc

def test_cnn(session, frame_spec_label_test, predictions, accuracy, ph_action, ph_label, ph_train):
    # Splitting the Data into Smaller Parts to avoid Memory Overflow in GPU
    # Testing the Trained Model on Test Data
    frame_spec_label_test = shuffle(frame_spec_label_test)
    splits = int(frame_spec_label_test.shape[0] / 1)
    batch = int(frame_spec_label_test.shape[0] / splits)
    accuracies = []
    prediction = []
    tst = time.time()
    for k in range(splits):
        batch_accuracy = session.run(accuracy, feed_dict={ph_action: frame_spec_label_test[batch*k:(batch*(k+1))+1, 0][0], ph_label: frame_spec_label_test[batch*k:(batch*(k+1))+1, 4][0]*np.ones(13).reshape(1, 13), ph_train: False})
        batch_prediction = session.run(predictions, feed_dict={ph_action: frame_spec_label_test[batch*k:(batch*(k+1))+1, 0][0], ph_train: False})
        accuracies.append(batch_accuracy)
        prediction.append(batch_prediction[0])
    total_test_acc = sum(accuracies) / len(accuracies)
    tet = time.time()
    print ('Test Accuracy : %f ::: Testing Time : %f Secs.' % (total_test_acc, (tet - tst)))
    # Return Test Accuracy and Predictions
    return total_test_acc, prediction

def run_cnn(config, frame_spec_label_train, frame_spec_label_valid, frame_spec_label_test, predictions, accuracy, ph_action, ph_label, ph_train):
    # Create TensorFlow Session with GPU setting.
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
            
        # Train using The Train Dataset Split
        frame_spec_label_train = shuffle(frame_spec_label_train)
        splits = int(frame_spec_label_train.shape[0] / 1)
        batch = int(frame_spec_label_train.shape[0] / splits)
        accuracies = []
        tstr = time.time()
        for s in range(splits):
            batch_accuracy = sess.run(accuracy, feed_dict={ph_action: frame_spec_label_train[batch*s:(batch*(s+1))+1, 0][0], ph_label: frame_spec_label_train[batch*s:(batch*(s+1))+1, 4][0]*np.ones(13).reshape(1, 13), ph_train: True})
            accuracies.append(batch_accuracy)
        total_train_acc = sum(accuracies) / len(accuracies)
        tetr = time.time()
        print ('Training Accuracy : %f ::: Training Time : %f Secs.' % ((total_train_acc), (tetr - tstr)))
            
        # Splitting the Data into Smaller Parts to avoid Memory Overflow in GPU
        # Testing the Trained Model on Validation Dataset Split
        frame_spec_label_valid = shuffle(frame_spec_label_valid)
        splits = int(frame_spec_label_valid.shape[0] / 1)
        batch = int(frame_spec_label_valid.shape[0] / splits)
        accuracies = []
        tsv = time.time()
        for j in range(splits):
            batch_accuracy = sess.run(accuracy, feed_dict={ph_action: frame_spec_label_valid[batch*j:(batch*(j+1))+1, 0][0], ph_label: frame_spec_label_valid[batch*j:(batch*(j+1))+1, 4][0]*np.ones(13).reshape(1, 13), ph_train: False})
            accuracies.append(batch_accuracy)
        total_valid_acc = sum(accuracies) / len(accuracies)
        tev = time.time()
        print ('Validation Accuracy : %f ::: Validation Time : %f Secs.' % (total_valid_acc, (tev - tsv)))
            
        # Splitting the Data into Smaller Parts to avoid Memory Overflow in GPU
        # Testing the Trained Model on Test Data
        frame_spec_label_test = shuffle(frame_spec_label_test)
        splits = int(frame_spec_label_test.shape[0] / 1)
        batch = int(frame_spec_label_test.shape[0] / splits)
        accuracies = []
        prediction = []
        tst = time.time()
        for k in range(splits):
            batch_accuracy = sess.run(accuracy, feed_dict={ph_action: frame_spec_label_test[batch*k:(batch*(k+1))+1, 0][0], ph_label: frame_spec_label_test[batch*k:(batch*(k+1))+1, 4][0]*np.ones(13).reshape(1, 13), ph_train: False})
            batch_prediction = sess.run(predictions, feed_dict={ph_action: frame_spec_label_test[batch*k:(batch*(k+1))+1, 0][0], ph_train: False})
            accuracies.append(batch_accuracy)
            prediction.append(batch_prediction[0])
        total_test_acc = sum(accuracies) / len(accuracies)
        tet = time.time()
        print ('Test Accuracy : %f ::: Testing Time : %f Secs.' % (total_test_acc, (tet - tst)))
    
    # Return The Train Validation Test Accuracies
    return total_train_acc, total_valid_acc, total_test_acc, prediction

def calc_metrics(labels, preds):
    # Calculate Evaluation Metrics
    true, false = 0, 0
    for l in range(len(labels)):
        if (labels[l] == preds[l]):
            true += 1
        else:
            false += 1
    sensitivity = float(true / len(labels))
    specificity = float(false / len(labels))
    print ('Test Sensitivity : %f ::: Test Specificity : %f' % (sensitivity, specificity))
    print ('~' * 90)
    # Return Metrics
    return sensitivity, specificity

def run_cnn_wout_epochs(max_epochs, fsl_train, fsl_valid, fsl_test):
    # Run the CNN Model for specified Epochs
    for n in range(max_epochs):
        print ('Epoch : ' + str(n + 1))
        with tf.Graph().as_default():
            ph_action, ph_label, ph_train = set_placeholders()
            final_layer, dense_units = my_cnn(ph_action, ph_train)
            logits = get_logits(final_layer, dense_units)
            loss = get_loss(ph_label, logits)
            train_op = get_train_op(loss, 0.001)
            accuracy = get_accuracy(ph_label, logits)
            predictions = get_predictions(logits)
            config = set_gpu_props()
            train_acc, valid_acc, test_acc, predicts = run_cnn(config, fsl_train, fsl_valid, fsl_test, predictions, accuracy, ph_action, ph_label, ph_train)
            sensitivity, specificity = calc_metrics(fsl_test[:, 4], predicts)
#            if (((train_acc > 0.90) and (valid_acc > 0.90)) or (test_acc > 0.90)):
#                break
    return None

def run_cnn_epochs(train_epochs, fsl_train, fsl_valid, fsl_test):
    # Run The CNN model for specified Training Epochs
    with tf.Graph().as_default():
        ph_action, ph_label, ph_train = set_placeholders()
        final_layer, dense_units = my_cnn(ph_action, ph_train)
        logits = get_logits(final_layer, dense_units)
        loss = get_loss(ph_label, logits)
        train_op = get_train_op(loss, 0.001)
        accuracy = get_accuracy(ph_label, logits)
        predictions = get_predictions(logits)
        config = set_gpu_props()
        sess = init_sess(config)
        # Train
        for e in range(train_epochs):
            print ('Train Epoch : ' + str(e + 1))
            train_acc = train_cnn(sess, fsl_train, accuracy, ph_action, ph_label, ph_train)
            # Make use of train_acc here if needed.
        # Validate
        valid_acc = validate_cnn(sess, fsl_valid, accuracy, ph_action, ph_label, ph_train)
        # Test
        test_acc, test_preds = test_cnn(sess, fsl_test, predictions, accuracy, ph_action, ph_label, ph_train)
        # Calculate Test Sensitivity and Test Specificity
        sensitivity, specificity = calc_metrics(fsl_test[:, 4], test_preds)
#        if (((train_acc > 0.90) and (valid_acc > 0.90)) or (test_acc > 0.90)):
#            break
    return test_preds

def plot_preds(test_labels, test_preds):
    # Plot Test Labels vs. Predictions
    plt.figure('Labels vs. Predictions')
    plt.plot(test_labels, 'b.')
    plt.plot(test_preds, 'r.')
    plt.xlabel('Frames')
    plt.ylabel('Class')
    plt.legend(['Labels', 'Predictions'], loc='best')
    return None


# Get Datasets
print ('~' * 90)
train_data, valid_data, test_data = get_dataset_splits()
fsl_train = get_videos_as_images_with_labels(train_data, 'Train') # ~4.00 Mins
fsl_valid = get_videos_as_images_with_labels(valid_data, 'Validation') # ~0.50 Mins
fsl_test = get_videos_as_images_with_labels(test_data, 'Test') # ~0.50 Mins
print ('~' * 90)

# Get Predictions
test_preds = run_cnn_wout_epochs(1, fsl_train, fsl_valid, fsl_test)
#test_preds = run_cnn_epochs(3, fsl_train, fsl_valid, fsl_test)

# plot Predictions
plot_preds(fsl_test[:, 4], test_preds)


# End of File