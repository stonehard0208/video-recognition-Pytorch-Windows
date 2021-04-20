import csv
import glob
import os
import os.path
from subprocess import call
import ffmpy


# 从路径获取classname,filename
def get_video_name (path):
    #file = path.split ("\\")  # 将./train\ApplyEyeMakeup\v_ApplyEyeMakeup_g08_c01.avi 按照\分开

    #filename = file [2]  # v_ApplyEyeMakeup_g08_c01.avi
    #filename_no_dot = filename.split ('.') [0]  # v_ApplyEyeMakeup_g08_c01
    #classname = file [1]  # ApplyEyeMakeup
    #train_test = file [0]  # ./train
    #train_test = file [0].split ("/") [1]  # train
    #return train_test, classname, filename_no_dot, filename


    file = path.split ("\\")  # 将./train\ApplyEyeMakeup\v_ApplyEyeMakeup_g08_c01.avi 按照\分开

    filename = file [1]  # v_ApplyEyeMakeup_g08_c01.avi
    filename_no_dot = filename.split ('.') [0]  # v_ApplyEyeMakeup_g08_c01
    classname = file [1]  # ApplyEyeMakeup
    train_test = file [0]  # ./train
    train_test = file [0].split ("/") [1]  # train
    return train_test, classname, filename_no_dot, filename


# 确认是否存在图片
def if_convert (path):
    train_test, classname, filename_no_dot, filename = path

    return bool (os.path.exists (train_test + '/' + classname +
                                 '/' + '-0001.jpg'))


# 创建文件夹
def mkdir (path):
    folder = os.path.exists (path)
    if not folder:
        os.makedirs (path)
        # print(path + "OK")
    # else:
    #    print(path+"No")
    return path


# 将视频转换为图片
def convert ():
    folders = ['./hmdb51_org']

    for file in folders:
        # 搜索匹配
        search_folders = glob.glob (file + '*')

        for video in search_folders:
            avi_file = glob.glob (video + '/*.avi')

            for video_path in avi_file:
                path = get_video_name (video_path)
                train_test, classname, filename_no_dot, filename = path

                if not if_convert (path):
                    src = train_test + '/' + classname + '/' + \
                          filename
                    jpg_path = train_test + '/' + classname + '/'
                    jpg_dir = mkdir (jpg_path)
                    dest = jpg_dir + filename_no_dot + '_' + '-%000001d.jpg'
                    call (["ffmpeg", "-i", src, "-r", "1", dest], shell=True)


convert ()
