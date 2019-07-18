import os
#import numpy as np
pat_1 = "/storage/research/Intern19_v2/AutomatedDetectionWSI/extract/"
pat_2 = "/storage/research/Intern19_v2/AutomatedDetectionWSI/extract2/"

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

ext = os.listdir(pat_1)
ext2 = os.listdir(pat_2)
ext = sorted(ext)
ext2= sorted(ext2)

def save_file2csv(file_dir, file_name):
    """
    save file path to csv
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    
    for i in zip(ext,ext2):
        print(i)
        #ex = os.walk(os.path.join(pat_1,i[0]))
        #ex = sorted(list(ex)[0][2])
        #ex2 = os.walk(os.path.join(pat_2,i[1]))
        #ex2 = sorted(list(ex2)[0][2]) 
    
        file_image_dir = file_dir + "/"+ 'extract' + i[0]
        file_mask_dir = file_dir + "/" + 'extract2'+ i[1]
        file_paths = file_name_path(file_image_dir, dir=False, file=True)
        out.writelines("Image,Mask" + "\n")
        for index in range(len(file_paths)):
            out_file_image_path = file_image_dir + "/" + file_paths[index]
            out_file_mask_path = file_mask_dir + "/" + file_paths[index]
            out.writelines(out_file_image_path + "," + out_file_mask_path + "\n")


save_file2csv("/storage/research/Intern19_v2/AutomatedDetectionWSI/", "trainnpy.csv")
