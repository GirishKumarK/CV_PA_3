import os
import sys
import time
import random
import numpy as np
import pandas as pd
from sklearn import svm
import skimage.io as skiio
import skvideo.io as skvio
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut


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

def get_dataset():
    # A Method to Read the Raw Dataset and Return Dataset As Is
    # Get Dataset
    dataset = get_data_info()
    # Convert to NumPy Array
    dataset = np.array(dataset)
    # Convert to List of Lists
    dataset = dataset.tolist()
    # Shuffle Dataset To Randomize the Ordered Labels
    dataset = shuffle(dataset)
    # Return the DataFrame Splits of Train and Test
    dataset = pd.DataFrame(dataset, columns=['Action', 'Label'])
    return dataset

def get_loocv_splits(dataset):
    # A Method That Returns A List Of Lists Of Train and Test LeaveOneOut Splits
    dataset = np.array(dataset)
    # Initialize LOO CV Method
    loo = LeaveOneOut()
    loo_data_splits = []
    for train_index, test_index in loo.split(dataset):
        train, test = dataset[train_index], dataset[test_index]
        loo_data_splits.append([train.tolist(), test.tolist()])
    # Convert NumPy Array to Pandas DataFrame
    loo_data_splits = pd.DataFrame(loo_data_splits, columns=['Train', 'Test'])
    # Return LOOCV Splits List
    return loo_data_splits

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
    # Normalizing All Video Frame Sizes to h120 x w160 [Ratio 3:4]
    h, w = 90, 120
    dim = str(w) + 'x' + str(h)
    # Read Video As GrayScale
    video = skvio.vread(video_file, as_grey=True, outputdict={'-sws_flags': 'bilinear', '-s': dim}) # -s : width x height
    frames, height, width, channels = video.shape
    # Normalize Video Frame Intensities
    if (channels > 1):
        for f in range(frames):
            video[f] = normalize_image(video[f])
    # Return Normalized Video and Specs
    return video, frames, height, width, channels, video_file_name

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
        images, frames, height, width, channels, vidname = get_video_specs(videos[vdo][0])
        for img in images:
            image = img.reshape(height, width, channels)
            frame_spec_label.append([image, height, width, channels, labels_encoded[vdo], vidname])
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

def get_hog_features(images, labels, vdo_name, data_name):
    # Get a List of all HOG Features for given Video Frames
    hog_feats = []
    tshog = time.time()
    for img in range(len(images)):
        # Get only the HOG Features
        _, features, _ = get_hog(images[img])
        hog_feats.append([features.tolist(), labels[img], vdo_name[img]])
    tehog = time.time()
    print ('Time Taken To Get %s Data HOG Features List : %f Mins.' % (data_name, ((tehog - tshog) / 60.0)))
    # Return the List of HOG Features
    return hog_feats

def get_hog_train_test_splits(loo_splits, hog_data):
    # A Method to Obtain All Train and Test HOG Features Lists
    tshs = time.time()
    tests = np.array([row['Test'] for index, row in loo_splits.iterrows()]).tolist()
    hog_data = np.array(hog_data)
    hog_trains, hog_tests = [], []
    for test in tests:
        # Get Video Name Frame Match In HOG Features List
        hog_test_index = np.where(hog_data[:, 2] == test[0][0])[0].tolist()
        hog_train_index = np.where(hog_data[:, 2] != test[0][0])[0].tolist()
        hog_test = []
        hog_train = []
        for it in hog_test_index:
            hog_test.append(hog_data[it].tolist())
        for itr in hog_train_index:
            hog_train.append(hog_data[itr].tolist())
        hog_tests.append(hog_test)
        hog_trains.append(hog_train)
    tehs = time.time()
    print ('Time Taken To Get Train and Test HOG Features Splits : %f Secs.' % (tehs - tshs))
    # Return List of HOG Train and Test Lists
    return hog_trains, hog_tests

def flatten_hog_splits(hog_train, hog_test):
    # A Method To Flatten HOG Train and Test Features Lists
    # Trains
    hog_train_flat = []
    for video in hog_train:
        feats = video[0]
        label = video[1]
        hog_train_flat.append([feats, label])
    # Tests
    hog_test_flat = []
    for video in hog_test:
        feats = video[0]
        label = video[1]
        hog_test_flat.append([feats, label])
    # Convert To Pandas DataFrame
    hog_train_flat = pd.DataFrame(hog_train_flat, columns=['Feature', 'Label'])
    hog_test_flat = pd.DataFrame(hog_test_flat, columns=['Feature', 'Label'])
    # Get the List of Train and Test Frame Features
    train_feats = np.array([row['Feature'] for index, row in hog_train_flat.iterrows()])
    test_feats = np.array([row['Feature'] for index, row in hog_test_flat.iterrows()])
    # Get the List of Labels of Train and Test Frames
    train_labels = np.array([row['Label'] for index, row in hog_train_flat.iterrows()]).ravel()
    test_labels = np.array([row['Label'] for index, row in hog_test_flat.iterrows()]).ravel()
    # Return Flattened HOG Train and Test Features List
    return train_feats, test_feats, train_labels, test_labels

def do_svm_train(kernel, train_data, train_labels):
    # Convert Datset to Numpy Array
    train_data = np.array(train_data)
    train_labels = np.array([trl for trl in train_labels]).ravel()
    # Kernel Constants
    lin_c = 1.0
    rbf_c, rbf_gamma = 1.0, 'auto'
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
        kernel_fit = svc_lin
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
        kernel_fit = svc_rbf
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
        kernel_fit = svc_sgm
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
        kernel_fit = svc_poly
    # Return SVM Kernel Fit Models
    return kernel_fit

def do_svm_test(kernel_fit, test_data, test_labels):
    # Perform SVM Predictions for Specified Kernel
    # Convert Datset to Numpy Array
    test_data = np.array(test_data)
    test_labels = np.array([tl for tl in test_labels]).ravel()
    # Test
    t = time.time()
    test_preds = kernel_fit.predict(test_data)
    svc_predict = time.time() - t
    print('Time Taken to Predict using the Fitted Model : ' + str(svc_predict) + ' Secs')
    accuracy = r2_score(test_labels, test_preds) * 100
    print('Test Accuracy Score of the Model : ' + str(accuracy))
    # Return Test Predictions
    return test_preds, accuracy
    

def get_plots(kernel, labels, predictions):
    # Plot Labels vs. Predictions
    if (kernel == 'linear'):
        # Compare Linear Labels vs. Predictions
        plt.figure('Linear Kernel SVM ~ SKLearn')
        plt.plot([x for x in range(0, len(labels))], labels, 'b.')
        plt.plot([x for x in range(0, len(labels))], predictions, 'r.')
        plt.title('Linear Kernel SVM ~ Labels vs. Predictions')
        plt.legend(['Labels', 'Predictions'])
    elif (kernel == 'gaussian'):
        # Compare Gaussian Labels vs. Predictions
        plt.figure('Gaussian Kernel SVM ~ SKLearn')
        plt.plot([x for x in range(0, len(labels))], labels, 'b.')
        plt.plot([x for x in range(0, len(labels))], predictions, 'r.')
        plt.title('Gaussian Kernel SVM ~ Labels vs. Predictions')
        plt.legend(['Labels', 'Predictions'])
    elif (kernel == 'sigmoid'):
        # Compare Sigmoid Labels vs. Predictions
        plt.figure('Sigmoid Kernel SVM ~ SKLearn')
        plt.plot([x for x in range(0, len(labels))], labels, 'b.')
        plt.plot([x for x in range(0, len(labels))], predictions, 'r.')
        plt.title('Sigmoid Kernel SVM ~ Labels vs. Predictions')
        plt.legend(['Labels', 'Predictions'])
    elif (kernel == 'poly'):
        # Compare Polynomial Labels vs. Predictions
        plt.figure('Polynomial Kernel SVM ~ SKLearn')
        plt.plot([x for x in range(0, len(labels))], labels, 'b.')
        plt.plot([x for x in range(0, len(labels))], predictions, 'r.')
        plt.title('Polynomial Kernel SVM ~ Labels vs. Predictions')
        plt.legend(['Labels', 'Predictions'])
    # Return Nothing
    return None

def calc_metrics(labels, predictions):
    # Calculate Evaluation Metrics
    true, false = 0.0, 0.0
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


def do_hogsvm_loocv():
    # Function to Execute HOG SVM Program
    
    # Get Dataset
    print ('~' * 90)
    dataset = get_dataset()
    print ('~' * 90)
    
    # Get LOOCV Splits
    loo_splits = get_loocv_splits(dataset)
    loo_length = loo_splits.shape[0]
    
    # Get Data Splits with Labels
    fsl_data = get_videos_as_images_with_labels(dataset, 'All') # ~5.00 Mins
    print ('~' * 90)
    
    # Get HOG Features List
    hog_data = get_hog_features(fsl_data[:, 0], fsl_data[:, 4], fsl_data[:, 5], 'All') # ~12.00 Mins
    print ('~' * 90)
    
    # Get HOG Features Train and Test Splits
    hog_trains, hog_tests = get_hog_train_test_splits(loo_splits, hog_data) # ~30.00 Secs
    print ('~' * 90)
    
    # Perform SVM on HOG Train and Test Features Splits using Specified Kernel
    accs, senss, specs = [], [], []
    for epoch in range(loo_length): # loo_length
        print ('Epoch : ' + str(epoch + 1))
        # Get Flattened Train and Test Features and Labels
        train_feats, test_feats, train_labels, test_labels = flatten_hog_splits(hog_trains[epoch], hog_tests[epoch])
        # Perform SVM on HOG Features using Specified Kernel
        # Linear
        lin_fit = do_svm_train('linear', train_feats, train_labels)
        lin_preds, lin_acc = do_svm_test(lin_fit, test_feats, test_labels)
#        get_plots('linear', test_labels, lin_preds)
        lin_sens, lin_spec = calc_metrics(test_labels, lin_preds)
        accs.append(lin_acc)
        senss.append(lin_sens)
        specs.append(lin_spec)
#        # Sigmoid
#        sgm_fit = do_svm_train('sigmoid', train_feats, train_labels)
#        sgm_preds, sgm_acc = do_svm_test(sgm_fit, test_feats, test_labels)
#        get_plots('sigmoid', test_labels, sgm_preds)
#        sgm_sens, sgm_spec = calc_metrics(test_labels, sgm_preds)
#        accs.append(sgm_acc)
#        senss.append(sgm_sens)
#        specs.append(sgm_spec)
#        # Polynomial
#        poly_fit = do_svm_train('poly', train_feats, train_labels)
#        poly_preds, poly_acc = do_svm_test(poly_fit, test_feats, test_labels)
#        get_plots('poly', test_labels, poly_preds)
#        poly_sens, poly_spec = calc_metrics(test_labels, poly_preds)
#        accs.append(poly_acc)
#        senss.append(poly_sens)
#        specs.append(poly_spec)
#        # Gaussian
#        rbf_fit = do_svm_train('gaussian', train_feats, train_labels)
#        rbf_preds, rbf_acc = do_svm_test(rbf_fit, test_feats, test_labels)
#        get_plots('gaussian', test_labels, rbf_preds)
#        rbf_sens, rbf_spec = calc_metrics(test_labels, rbf_preds)
#        accs.append(rbf_acc)
#        senss.append(rbf_sens)
#        specs.append(rbf_spec)
    avg_acc = sum(accs) / len(accs)
    avg_sens = sum(senss) / len(senss)
    avg_spec = sum(specs) / len(specs)
    print ('Average Accuracy : %f ::: Average Sensitivity : %f ::: Average Specificity :%f' % (avg_acc, avg_sens, avg_spec))
    print ('~' * 90)
    
    # Return None
    return None

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
    hog_train = get_hog_features(fsl_train[:, 0], fsl_train[:, 4], fsl_train[:, 5], 'Train') # ~9.00 Mins
    hog_test = get_hog_features(fsl_test[:, 0], fsl_test[:, 4], fsl_test[:, 5], 'Test') # ~1.00 Mins
    print ('~' * 90)
    
    # Perform SVM on HOG Features using Specified Kernel
    # Linear
    lin_fit = do_svm_train('linear', hog_train, fsl_train[:, 4])
    lin_preds, lin_acc = do_svm_test(lin_fit, hog_test, fsl_test[:, 4])
    get_plots('linear', fsl_test[:, 4], lin_preds)
    lin_sens, lin_spec = calc_metrics(fsl_test[:, 4], lin_preds)
#    # Sigmoid
#    sgm_fit = do_svm_train('sigmoid', hog_train, fsl_train[:, 4])
#    sgm_preds, sgm_acc = do_svm_test(sgm_fit, hog_test, fsl_test[:, 4])
#    get_plots('sigmoid', fsl_test[:, 4], lin_preds)
#    sgm_sens, sgm_spec = calc_metrics(fsl_test[:, 4], sgm_preds)
#    # Polynomial
#    poly_fit = do_svm_train('poly', hog_train, fsl_train[:, 4])
#    poly_preds, poly_acc = do_svm_test(poly_fit, hog_test, fsl_test[:, 4])
#    get_plots('poly', fsl_test[:, 4], poly_preds)
#    poly_sens, poly_spec = calc_metrics(fsl_test[:, 4], poly_preds)
#    # Gaussian
#    rbf_fit = do_svm_train('gaussian', hog_train, fsl_train[:, 4])
#    rbf_preds, rbf_acc = do_svm_test(rbf_fit, hog_test, fsl_test[:, 4])
#    get_plots('gaussian', fsl_test[:, 4], rbf_preds)
#    rbf_sens, rbf_spec = calc_metrics(fsl_test[:, 4], rbf_preds)
#    print ('~' * 90)
    
    # Return None
    return None


# End of File