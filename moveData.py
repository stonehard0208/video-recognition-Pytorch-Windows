import os
import os.path
from collections import Counter

#读取trainlist 和 testlist，并以list的形式存放在group中
def group_list(version):
    test_file = './ucfTrainTestlist/testlist' + version + '.txt'
    train_file = './ucfTrainTestlist/trainlist' + version + '.txt'


    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]  # strip 去掉空格
        # print(test_list)


    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list] # train_file中的‘ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1’ 按空格分开，取前面的，即.avi为结尾
        #print(train_list)

    print(train_list)
    print(test_list)

    group = {
        'train':train_list,
        'test':test_list
    }
    return group

#在train test中创建文件夹，并移动视频到文件夹中
def move_data(group):


    #print(group)
    for file,videos in group.items():
        #print(file)
        #print(videos)


        for real_video in videos:
        #print(video)
            name = real_video.split('/')
            #print(name)
            classname = name[0]
            filename = name[1]


            if not os.path.exists(file + '/' + classname):
                print("Creating folder for %s/%s" % (file, classname))
                os.makedirs(file + '/' + classname)



            if not os.path.exists('./' + classname + '/' + filename):
                print('/' + classname + '/' + filename)
                print("Can't find %s." % (filename))
                continue

            dest = file + '/' + classname + '/' + filename
            #print(dest)
            original = os.path.join(classname, filename)
            print("Moving %s to %s" % (filename, dest))
            os.rename(original, dest)



    print("Finished")

#删除空文件夹
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
    print("Finished")


group = group_list("01")
move_data(group)
delete_none_dir('../data/')
