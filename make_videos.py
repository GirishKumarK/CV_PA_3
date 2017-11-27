import os
import numpy as np
import skimage.io as skiio
import skvideo.io as skvio


def make_videos():
    # Function To Create All Videos From JPG Files
    
    # Main Directory
    main_dir = '.\\ucf_sports_actions\\ucf action'
    
    # Actions Directory
    axns_dir = os.listdir(main_dir)
    
    # Indicate Video Creation Process
    print ('~' * 100)
    print ('Creating Videos For Main Directory : ' + main_dir)
    
    # For All Root Directories
    for ad in axns_dir:
        
        # Get Current Action Directory
        curr_axn_dir = os.path.join(main_dir, ad)
        
        # Get Current Action Set Directory
        axns_set = os.listdir(curr_axn_dir)
        
        # Indicate Video Creation Process
        print ('~' * 100)
        print ('Creating Videos For Action Directory : ' + curr_axn_dir)
        
        # Generate Videos for All Action Sets in the Directory
        for aset in axns_set:
            current_dir = os.path.join(curr_axn_dir, aset)
            files = os.listdir(current_dir)
            
            # Get a list of all JPG files
            jpgs = []
            for file in files:
                if not os.path.isdir(file):
                    if file.endswith('.jpg'):
                        jpgs.append(os.path.join(current_dir, file))
            
            # A Flag to Indicate No JPG Files
            jpg_flag = False
            
            # Create Video for All Folders with JPG files
            if (len(jpgs) != 0):
                jpg_flag = True
                
                # Indicate Video Creation Process
                print ('~' * 100)
                print ('Generating Video For Action Root Directory : ' + current_dir)
                
                # Open Images in JPG list
                images, img_dims = [], []
                for jpg in jpgs:
                    image = skiio.imread(jpg)
                    height, width, channels = image.shape
                    images.append(image)
                    img_dims.append([height, width, channels])
                images = np.array(images)
                img_dims = np.array(img_dims)
                
                # Merge the Images to one AVI file
                video_name = (jpgs[0].split('.')[1].split('\\')[-1] + '.avi')
                video_path = os.path.join(current_dir, video_name)
                skvio.vwrite(video_path, images, inputdict={'-r':'10'})
                
                # Indicate Video Generation Completion
                print ('Video Created For Action Root Directory : ' + current_dir)
                
        if (jpg_flag == True):
            # Indicate Video Generation Completion
            print ('~' * 100)
            print ('Videos Created For Action Directory : ' + curr_axn_dir)
        else:
            # Indicate No JPG Files Found
            print ('~' * 100)
            print ('No JPG Files Found In Action Directory : ' + curr_axn_dir)
            print ('No Videos Created For Action Directory : ' + curr_axn_dir)
    
    # Indicate Video Generation Completion
    print ('~' * 100)
    print ('All Videos Generated Successfully !')
    print ('~' * 100)
    
    # Return Nothing
    return None


# End of File