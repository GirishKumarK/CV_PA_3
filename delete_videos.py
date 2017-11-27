import os
from shutil import rmtree

# Get Dataset Reference Location
for dirName, subdirList, fileList in os.walk('.', topdown=True):
    for fname in fileList:
        if (fname == 'action_label.csv'):
            # Delete Directory
            rmtree(dirName)

print ('~' * 100)
print ('Dataset Reference Directory Deleted Successfully !')
print ('~' * 100)


# End of File