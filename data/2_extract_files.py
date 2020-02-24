"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call
import ffmpy


def extract_files():
    """After we have all of our videos split between train and test, and
    all nested within folders representing their classes, we need to
    make a data file that we can reference when training our RNN(s).
    This will let us keep track of image sequences and other parts
    of the training process.

    We'll first need to extract images from each of the videos. We'll
    need to record the following data in the file:

    [train|test], class, filename, nb frames

    Extracting can be done with ffmpeg:
    `ffmpeg -i video.mpg image-%04d.jpg`
    """
    data_file = []
    folders = ['./train/', './test/']

    for folder in folders:
        class_folders = glob.glob(folder + '*') #glob的作用是匹配（搜索）
        #print(class_folders)
        for vid_class in class_folders:
            #print(vid_class)
            class_files = glob.glob(vid_class + '/*.avi')
            #print(class_files)
            for video_path in class_files:
                print(video_path)
                # Get the parts of the file.
                video_parts = get_video_parts(video_path)
                #获得字符串
                train_or_test, classname, filename_no_ext, filename = video_parts

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                if not check_already_extracted(video_parts):
                    # Now extract it.
                    src = train_or_test + '/' + classname + '/' + \
                        filename
                    jpg_path = train_or_test + '/' + classname + '/'
                    jpg_dir = mkdir(jpg_path)
                    dest = jpg_dir + filename_no_ext + '_' +  '-%000001d.jpg'
                    call(["ffmpeg", "-i", src, "-r" , "1" ,dest],shell=True)

                # Now get how many frames it is.
                nb_frames = get_nb_frames_for_video(video_parts)

                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(train_or_test + '/' + classname + '/' +
                                  '*.jpg')
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split("\\") #将./train\ApplyEyeMakeup\v_ApplyEyeMakeup_g08_c01.avi 按照\分开
    #print(parts[1])
    filename = parts[2] #v_ApplyEyeMakeup_g08_c01.avi
    filename_no_ext = filename.split('.')[0] #v_ApplyEyeMakeup_g08_c01
    classname = parts[1] #ApplyEyeMakeup
    train_or_test = parts[0] #./train
    #print(train_or_test)
    train_or_test = parts[0].split("/")[1]
    #print(train_or_test)

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(train_or_test + '/' + classname +
                               '/' +  '-0001.jpg'))

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        # print(path + "OK")
    #else:
    #    print(path+"No")
    return path


def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    extract_files()

if __name__ == '__main__':
    main()
