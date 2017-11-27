import os
import sys
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import svm
from shutil import copy2
from shutil import rmtree
import skvideo.io as skvio
import skimage.io as skiio
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import r2_score

from make_videos import make_videos
from collect_videos import collect_videos
from delete_videos import delete_videos

from cv_pa3_CNN import do_cnn
from cv_pa3_HOG_SVM import do_hogsvm

def first_run():
    # Create the Datasets for First Run
    make_videos()
    time.sleep(2.0)
    collect_videos()
    print ('Datasets Created Successfully !')
    print ('~' * 100)
    return None

print ('~' * 110)
print ('Attention : \n Program Requirements : NumPy, Pandas, SciKit-Learn, SciKit-Image, SciKit-Video, MatPlotLib and TensorFlow')
print ('~' * 110)

firstrun = input('Are You Executing This Program For The First Time On This Computer ? : \n [Y] Yes \n [N] No \n Enter Character : ')
if ((firstrun == 'Y') or (firstrun == 'y')):
    print ('~' * 100)
    print ('Creating Datasets ...')
    time.sleep(1.0)
    first_run()
elif ((firstrun == 'N') or (firstrun == 'n')):
    print ('~' * 100)
    print ('Using Already Created Datasets')
    print ('~' * 100)
else:
    sys.exit('Wrong Choice ! Re-Execute Program !')

program = input('Choose Program to Execute : \n 1. HOG~SVM [SciKit] \n 2. CNN [TensorFlow] \n Enter Number : ')
if (program == '1'):
    print ('~' * 100)
    print ('Executing HOG~SVM Program ...')
    do_hogsvm()
    print ('Program Execution Complete !')
    print ('~' * 100)
elif (program == '2'):
    print ('~' * 100)
    print ('Executing CNN Program ...')
    do_cnn()
    print ('Program Execution Complete !')
    print ('~' * 100)
else:
    print ('~' * 100)
    print ('Wrong Choice ! Re-Execute Program !')
    print ('~' * 100)

destroy = input('Please Destroy Dataset Directory Once Complete ! \n [Y] Destroy \n [N] Do Not Destroy \n Enter Character : ')
if ((destroy == 'Y') or (destroy == 'y')):
    print ('~' * 100)
    print ('Dataset Directory Destroyed ! Please Re-Create In The Next Run !')
    print ('~' * 100)
    time.sleep(3.0)
    delete_videos()
else:
    print ('~' * 100)
    print ('Dataset Directory Not Destroyed ! To Destroy : Run Command \'delete_videos()\' In Console !')
    print ('~' * 100)
    


# End of File