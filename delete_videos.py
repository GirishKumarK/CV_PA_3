import os
import sys
from shutil import rmtree


def delete_videos():
    # Function to Delete the Created Dataset Directory
    
    # Get Dataset Reference Location
    for dirName, subdirList, fileList in os.walk('.', topdown=True):
        for fname in fileList:
            if (fname == 'action_label.csv'):
                # Delete Directory
                rmtree(dirName)
    
    # Exit System
    sys.exit('Program Terminated !')
    
    # Return None
    return None


# End of File