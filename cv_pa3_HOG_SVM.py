import os
import sys
import time
import random
import numpy as np
import pandas as pd
from sklearn import svm
import skimage.io as skiio
import skvideo.io as skvio
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


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
    # Merge The Action Label Pairs as Train Dataset
    train_data = []
    for a in actions:
        for v in a:
            # Generate Action Label Pair and Add to Train Dataset
            train_data.append([videos[v], labels[v]])
    # Shuffle Train Data To Randomize the Ordered Labels
    train_data = shuffle(train_data)
    # Return the DataFrame Splits of Train and Test
    train_data = pd.DataFrame(train_data, columns=['Action', 'Label'])
    test_data = pd.DataFrame(test_data, columns=['Action', 'Label'])
    return train_data, test_data

def get_shuffled_data(dataset):
    # Shuffle the Dataset to Randomize the Ordered Labels
    shuffled_data = shuffle(dataset)
    return shuffled_data

def normalize_image(image):
    # Normalize All Video Frame Intensities
    intensities = image.ravel()
    mean_int = np.mean(intensities)
    std_int = np.std(intensities)
    for i in range(len(intensities)):
        intensities[i] = ((intensities[i] - mean_int) / std_int)
    new_image = intensities.reshape(image.shape)
    return new_image

def get_video_specs(video_file_name):
    # Search and Obtain Video from Video File Name
    vids_dir = get_data_ref().split('\\')
    vids_dir = os.path.join(vids_dir[0], vids_dir[1])
    video_file = os.path.join(vids_dir, video_file_name)
    # Normalizing All Video Frame Sizes to h120 x w160
    h, w = 120, 160
    dim = str(w) + 'x' + str(h)
    video = skvio.vread(video_file, as_grey=True, outputdict={'-sws_flags': 'bilinear', '-s': dim}) # -s : width x height
    frames, height, width, channels = video.shape
    # Normalize Video Frame Intensities
    if (channels > 1):
        for f in range(frames):
            video[f] = normalize_image(video[f])
    # Return Normalized Video and Specs
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
            image = img.reshape(height, width, channels)
            frame_spec_label.append([image, height, width, channels, labels_encoded[vdo]])
    frame_spec_label = np.array(frame_spec_label)
    tes = time.time()
    print ('Time Taken To Get %s Frame_Specs_Label List : %f Mins.' % (split_name, ((tes - tss) / 60.0)))
    return frame_spec_label

def get_hog(image):
    # Reshape the Incoming Video Frame
    image = image.reshape(image.shape[0], image.shape[1])
    # Get HOG Features and HOG Image
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), 
                              cells_per_block=(1, 1), block_norm='L2', visualise=True)
    # Return Features and HOG Image
    return image, features, hog_image

def visualize_hog(image, features, hog_image):
    # Plot Image and HOG Image for Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input Image')
    ax1.set_adjustable('box-forced')
    ax2.axis('off')
    ax2.imshow(hog_image, cmap='gray')
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    return None

def get_hog_features(images, data_name):
    # Get a List of all HOG Features for given Video Frames
    hog_feats = []
    tshog = time.time()
    for img in images:
        # Get only the HOG Features
        _, features, _ = get_hog(img)
        hog_feats.append(features.tolist())
    tehog = time.time()
    print ('Time Taken To Get %s Data HOG Features List : %f Mins.' % (data_name, ((tehog - tshog) / 60.0)))
    # Return the List of HOG Features
    return hog_feats

def do_svm(kernel, train_data, test_data, train_labels, test_labels):
    # Convert Datset to Numpy Array
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_labels = np.array([trl for trl in train_labels]).ravel()
    test_labels = np.array([tl for tl in test_labels]).ravel()
    # Kernel Constants
    lin_c = 0.1
    rbf_c, rbf_gamma = 1.0, 10.0
    sgm_c, sgm_gamma, sgm_coef = 1.0, 1.0, 1.0
    poly_c, poly_deg, poly_gamma, poly_coef = 1.0, 1.0, 1.0, 1.0
    # Perform SVM Classification for Specified Kernel
    if (kernel == 'linear'):
        #Linear
        print('\nLinear Kernel with Hyperparameters : C = ' + str(lin_c))
        # Train
        svc_lin = svm.SVC(kernel='linear', C=lin_c, probability=True, cache_size=4096)
        t = time.time()
        fit_lin = svc_lin.fit(train_data, train_labels)
        svc_fit = time.time() - t
        print('Time Taken to Fit the Model : ' + str(svc_fit) + ' Secs')
        # Test
        t = time.time()
        y_lin = svc_lin.predict(test_data)
        svc_predict = time.time() - t
        print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
        acc_lin = r2_score(test_labels, y_lin) * 100
        print('Test Accuracy Score of the Model : ' + str(acc_lin))
        # Compare Labels vs. Predictions
        plt.figure('Linear Kernel SVM ~ SKLearn')
        plt.plot([x for x in range(0, len(test_labels))], test_labels, 'b.')
        plt.plot([x for x in range(0, len(test_labels))], y_lin, 'r.')
        plt.title('Linear Kernel SVM ~ Labels vs. Predictions')
        plt.legend(['Labels', 'Predictions'])
        test_preds = y_lin
    elif (kernel == 'gaussian'):
        #Gaussian
        print('\nGaussian Kernel with Hyperparameters : C = ' + str(rbf_c) + 
              ', Gamma = ' + str(rbf_gamma))
        # Train
        svc_rbf = svm.SVC(kernel='rbf', C=rbf_c, gamma=rbf_gamma, probability=True, cache_size=4096)
        t = time.time()
        fit_rbf = svc_rbf.fit(train_data, train_labels)
        svc_fit = time.time() - t
        print('Time Taken to Fit the Model : ' + str(svc_fit) + ' Secs')
        # Test
        t = time.time()
        y_rbf = svc_rbf.predict(test_data)
        svc_predict = time.time() - t
        print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
        acc_rbf = r2_score(test_labels, y_rbf) * 100
        print('Test Accuracy Score of the Model : ' + str(acc_rbf))
        # Compare Labels vs. Predictions
        plt.figure('Gaussian Kernel SVM ~ SKLearn')
        plt.plot([x for x in range(0, len(test_labels))], test_labels, 'b.')
        plt.plot([x for x in range(0, len(test_labels))], y_rbf, 'r.')
        plt.title('Gaussian Kernel SVM ~ Labels vs. Predictions')
        plt.legend(['Labels', 'Predictions'])
        test_preds = y_rbf
    elif (kernel == 'sigmoid'):
        #Sigmoid
        print('\nSigmoid Kernel with Hyperparameters : C = ' + str(sgm_c) + 
              ', Gamma = ' + str(sgm_gamma) + ', Coeff = ' + str(sgm_coef))
        # Train
        svc_sgm = svm.SVC(kernel='sigmoid', C=sgm_c, gamma=sgm_gamma, coef0=sgm_coef, probability=True, cache_size=4096)
        t = time.time()
        fit_sgm = svc_sgm.fit(train_data, train_labels)
        svc_fit = time.time() - t
        print('Time Taken to Fit the Model : ' + str(svc_fit) + ' Secs')
        # Test
        t = time.time()
        y_sgm = svc_sgm.predict(test_data)
        svc_predict = time.time() - t
        print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
        acc_sgm = r2_score(test_labels, y_sgm) * 100
        print('Test Accuracy Score of the Model : ' + str(acc_sgm))
        # Compare Labels vs. Predictions
        plt.figure('Sigmoid Kernel SVM ~ SKLearn')
        plt.plot([x for x in range(0, len(test_labels))], test_labels, 'b.')
        plt.plot([x for x in range(0, len(test_labels))], y_sgm, 'r.')
        plt.title('Sigmoid Kernel SVM ~ Labels vs. Predictions')
        plt.legend(['Labels', 'Predictions'])
        test_preds = y_sgm
    elif (kernel == 'poly'):
        #Polynomial
        print('\nPolynomial Kernel with Hyperparameters : C = ' + str(poly_c) + 
              ', Degree = ' + str(poly_deg) + ', Gamma = ' + str(poly_gamma) + 
              ', Coeff = ' + str(poly_coef))
        # Train
        svc_poly = svm.SVC(kernel='poly', C=poly_c, degree=poly_deg, gamma=poly_gamma, coef0=poly_coef, probability=True, cache_size=4096)
        t = time.time()
        fit_poly = svc_poly.fit(train_data, train_labels)
        svc_fit = time.time() - t
        print('Time Taken to Fit the Model : ' + str(svc_fit) + ' Secs')
        # Test
        t = time.time()
        y_poly = svc_poly.predict(test_data)
        svc_predict = time.time() - t
        print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
        acc_poly = r2_score(test_labels, y_poly) * 100
        print('Test Accuracy Score of the Model : ' + str(acc_poly))
        # Compare Labels vs. Predictions
        plt.figure('Polynomial Kernel SVM ~ SKLearn')
        plt.plot([x for x in range(0, len(test_labels))], test_labels, 'b.')
        plt.plot([x for x in range(0, len(test_labels))], y_poly, 'r.')
        plt.title('Polynomial Kernel SVM ~ Labels vs. Predictions')
        plt.legend(['Labels', 'Predictions'])
        test_preds = y_poly
    # Return Test labels and Test Predictions
    return test_labels, test_preds

def calc_metrics(labels, predictions):
    # Calculate Evaluation Metrics
    true, false = 0, 0
    for l in range(len(labels)):
        if (labels[l] == predictions[l]):
            true += 1
        else:
            false += 1
    sensitivity = float(true / len(labels))
    specificity = float(false / len(labels))
    print ('Test Sensitivity : %f ::: Test Specificity : %f' % (sensitivity, specificity))
    print ('~' * 90)
    # Return Metrics
    return sensitivity, specificity


def do_hogsvm():
    # Function to Execute HOG SVM Program
    
    # Get Dataset
    print ('~' * 90)
    train, test = get_dataset_splits()
    print ('~' * 90)
    
    # Get Data Splits with Labels
    fsl_train = get_videos_as_images_with_labels(train, 'Train') # ~4.50 Mins
    fsl_test = get_videos_as_images_with_labels(test, 'Test') # ~0.50 Mins
    print ('~' * 90)
    
    # Get HOG Features List
    hog_train = get_hog_features(fsl_train[:, 0], 'Train') # ~9.00 Mins
    hog_test = get_hog_features(fsl_test[:, 0], 'Test') # ~1.00 Mins
    print ('~' * 90)
    
    # Perform SVM on HOG Features using Specified Kernel
    lin_labels, lin_preds = do_svm('linear', hog_train, hog_test, fsl_train[:, 4], fsl_test[:, 4])
    lin_sens, lin_spec = calc_metrics(lin_labels, lin_preds)
    sgm_labels, sgm_preds = do_svm('sigmoid', hog_train, hog_test, fsl_train[:, 4], fsl_test[:, 4])
    sgm_sens, sgm_spec = calc_metrics(sgm_labels, sgm_preds)
    poly_labels, poly_preds = do_svm('poly', hog_train, hog_test, fsl_train[:, 4], fsl_test[:, 4])
    poly_sens, poly_spec = calc_metrics(poly_labels, poly_preds)
    rbf_labels, rbf_preds = do_svm('gaussian', hog_train, hog_test, fsl_train[:, 4], fsl_test[:, 4])
    rbf_sens, rbf_spec = calc_metrics(rbf_labels, rbf_preds)
    print ('~' * 90)
    
    # Return None
    return None


# End of File