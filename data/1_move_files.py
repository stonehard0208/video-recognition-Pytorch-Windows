"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import os
import os.path

def get_train_test_lists(version='01'):
    """
    Using one of the train/test files (01, 02, or 03), get the filename
    breakdowns we'll later use to move everything.
    """
    # Get our files based on version.
    test_file = './ucfTrainTestlist/testlist' + version + '.txt'
    train_file = './ucfTrainTestlist/trainlist' + version + '.txt'

    # Build the test list.
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)] #strip 去掉空格
        #print(test_list)


    # Build the train list. Extra step to remove the class index.
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list] # train_file中的‘ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1’ 按空格分开，取前面的，即.avi为结尾
        #print(train_list)

    # Set the groups in a dictionary.
    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups

def move_files(file_groups):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    # Do each of our groups.
    for group, videos in file_groups.items():

        # Do each of our videos.
        # print(group)
        # print(videos)
        for video in videos:

            # Get the parts.
            parts = video.split('/')
            classname = parts[0]
            filename = parts[1]
            #print(filename)

            # Check if this class exists.
            if not os.path.exists(group + '/' + classname):
                print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(group + '/' + classname)

            # Check if we have already moved this file, or at least that it
            # exists to move.
            #print(classname + '/' + filename)
            if not os.path.exists('./' + classname + '/' + filename):
                #print('/' + classname + '/' + filename)
                #print("Can't find %s to move. Skipping." % (filename))
                continue

            # Move it.
            dest = group + '/' + classname + '/' + filename
            print(dest)
            original = os.path.join(classname,filename)
            print("Moving %s to %s" % (filename, dest))
            os.rename(original, dest)

    print("Done.")

def delete_none_dir(dir):
    if os.path.isdir(dir):
        for d in os.listdir(dir):
            if d.endswith("test"):
                #print("a")
                continue
            elif d.endswith("train"):
                #print("B")
                continue
            elif d.endswith("ucfTrainTestlist"):
                #print("c")
                continue
            elif d.endswith(".py"):
                print("this is pyfile")
                continue
            else:
                os.rmdir(d)
    print("finished")

def main():
    """
    Go through each of our train/test text files and move the videos
    to the right place.
    """
    # Get the videos in groups so we can move them.
    group_lists = get_train_test_lists()

    # Move the files.
    move_files(group_lists)

    # 删除空文件夹
    delete_none_dir('../data/')

if __name__ == '__main__':
    main()
