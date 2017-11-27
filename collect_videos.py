import os
from shutil import copy2
import numpy as np
import pandas as pd


def collect_videos():
    # Function to Collect All Videos From Actions Directory
    
    # Create an Action-Label List to Save the Video-Label Pairs
    axn_lbl_list = []
    
    # Create a Destination Directory for All Videos
    dst_dir = '.\\action_videos'
    if not os.path.exists(dst_dir):
        print ('~' * 100)
        print ('Creating Destination Directory ...')
        os.makedirs(dst_dir)
        print ('Destination Directory Created !')
        print ('Copying Video Files ...')
    else:
        print ('~' * 100)
        print ('Destination Directory Already Exists.')
        print ('Copying Video Files ...')
    
    # Create a Copy of All Given Videos into the Directory
    rootDir = '.\\ucf_sports_actions\\ucf action'
    for dirName, subdirList, fileList in os.walk(rootDir, topdown=True):
        for fname in fileList:
            if fname.endswith('.avi'):
                print ('~' * 100)
                src = os.path.join(dirName, fname)
                print ('Copying Video File From : ' + src)
                dst = os.path.join(dst_dir, fname)
                print ('Copying Video File To   : ' + dst)
                copy2(src, dst)
                print ('File Copied Successfully !')
                # fname = action and split = label
                axn_lbl_list.append([fname, src.split('\\')[3]])
    
    # Convert the Final Action-Label List to NumPy Array
    axn_lbl_list = np.array(axn_lbl_list).astype(str).reshape(len(axn_lbl_list), 2)
    df = pd.DataFrame(axn_lbl_list)
    csv_name = os.path.join(dst_dir, 'action_label.csv')
    # Save the Action-Label List as CSV File
    df.to_csv(csv_name, header=['Action', 'Label'], index=False)
    
    # Indicate Copy Process Completion
    print ('~' * 100)
    print ('All Video Files Copied Successfully !')
    print ('~' * 100)
    
    # Return None
    return None


# End of File