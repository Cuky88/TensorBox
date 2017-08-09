# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import json
import argparse
import numpy as np
import shutil

def gta5_read(path, file):
    with open(path+file, 'r') as reader:
        for line in reader:
            print(line)
            yield line

def createJson(phase):
    annodir = '../data/GTA5/Test/'
    files = [f for f in listdir(annodir) if isfile(join(annodir, f))]
    i = 0
    j = 0
    total = []

    num = 20

    for c, file in enumerate(files):
        if c < num:
            if file.split('.')[-1] == 'txt':
                dic = {}
                rec = []
                label = gta5_read(annodir, file)
                frame = file.split('_')[1][:-4] + '.jpg'
                dic['image_path'] = 'Test_' + frame
                for s in label:
                    if '%' not in s:
                        x1 = s.split(' ')[0]
                        y1 = s.split(' ')[1]
                        x2 = s.split(' ')[2]
                        y2 = s.split(' ')[3]
                        print('frame %s: x1 %s, y1 %s - x2 %s, y2 %s'%(frame, x1, y1, x2, y2))
                        i += 1
                        #rec.append({"x1": int(x1), "x2": int(x1)+int(x2), "y1": int(y1), "y2": int(y1)+int(y2)})
                        rec.append({"x1": int(x1), "x2": int(x2), "y1": int(y1), "y2": int(y2)})
                dic['rects'] = rec
                total.append(dic)
                j += 1

    print(i)
    print(j)

    with open('../data/GTA5/' + phase + '_boxes.json', 'w') as outfile:
        json.dump(total, outfile, indent=2)
        outfile.write('\n')

def createVal():
    # Randomly select ~1000 images from train set and make ist the val set
    valset = np.random.randint(19060, size=(1, 1000))
    valset = set(valset[0])
    print(len(valset))

    train = []
    train_new = []
    val = []
    filenames = []

    with open('../data/GTA5/train_boxes.json', 'r') as reader:
        data = json.load(reader)
        for f in data:
            train.append(f)

    flag = 0
    for i, file in enumerate(train):
        flag = 0
        for ind in valset:
            if i == ind:
                flag = 1
                val.append(file)
                filenames.append(file['image_path'].strip('/')[1])
        if not flag:
            train_new.append(file)

    # annodir_train = 'data/COD20K/annotations/train/'
    # files = [f for f in listdir(annodir_train) if isfile(join(annodir_train, f))]
    #
    # try:
    #     for name in filenames:
    #         for file in files:
    #             if name == file:
    #                 shutil.copy(annodir_train + file, 'data/COD20K/annotations/val')
    # except IOError as e:
    #     print("Unable to copy file. %s" % e)
    # except:
    #     print("Unexpected error!")

    with open('../data/GTA5/val_boxes.json', 'w') as outfile:
        json.dump(val, outfile, indent=2)
        outfile.write('\n')
    print(len(val))

    with open('../data/GTA5/train_new_boxes.json', 'w') as writer:
        json.dump(train_new, writer, indent=2)
        writer.write('\n')
    print(len(train_new))

def showBB(img, rects, name):
    for bb in rects:
        print(bb)
        img = cv2.rectangle(img, (bb['x1'], bb['y1']), (bb['x2'], bb['y2']), (255, 0, 0), 1)

    plt.imshow(img)
    plt.show()

def checkAnno(check):
    anno = []
    with open(check, 'r') as reader:
        data = json.load(reader)
        for f in data:
            anno.append(f)

    for f in anno:
        for bb in f['rects']:
            if bb['y1'] >= bb['y2']:
                print('Error y1 greater-equal y2 in %s'%f['image_path'])


def main():
    print("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default=None, type=str)
    parser.add_argument('--show', default=None, type=int)
    parser.add_argument('--check', default=None, type=str)
    args = parser.parse_args()

    if args.phase is not None:
        print("Calling createJson")
        createJson(args.phase)
    elif args.phase == 'val':
        createVal()
    elif args.check is not None:
        checkAnno(args.check)

    if args.show is not None:
        with open('../data/GTA5/train_boxes.json', 'r') as reader:
            data = json.load(reader)
            for i, img in enumerate(data):
                if i == args.show:
                    img_loaded = cv2.imread('../data/GTA5/' + img['image_path'])
                    print('../data/GTA5/' + img['image_path'])

                    showBB(img_loaded, img['rects'], img['image_path'])


if __name__ == '__main__':
    main()