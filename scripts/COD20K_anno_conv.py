# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import json
import argparse
import numpy as np
from collections import OrderedDict

def cod20k_read(path, file):
    with open(path+file, 'r') as reader:
        for line in reader:
            yield line

def createJson(phase):
    annodir_train = '../data/COD20K/annotations/' + phase + '/'
    files = [f for f in listdir(annodir_train) if isfile(join(annodir_train, f))]
    i = 0
    j = 0
    total = []

    for file in files:
        dic = {}
        rec = []
        label = cod20k_read(annodir_train, file)
        frame = file.split('.')[0] + '.jpg'
        dic['image_path'] = phase + '/' + frame
        for s in label:
            if '%' not in s:
                x1 = s.split(' ')[1]
                y1 = s.split(' ')[2]
                x2 = s.split(' ')[3]
                y2 = s.split(' ')[4]
                # print('frame %s: x1 %s, y1 %s - x2 %s, y2 %s'%(frame, x1, y1, x2, y2))
                i += 1
                rec.append({"x1": int(x1), "x2": int(x1)+int(x2), "y1": int(y1), "y2": int(y1)+int(y2)})
        dic['rects'] = rec
        total.append(dic)
        j += 1
    print(i)
    print(j)

    with open('../data/COD20K/' + phase + '_boxes.json', 'w') as outfile:
        json.dump(total, outfile, indent=2)
        outfile.write('\n')

def sort_img(json):
    try:
        return json['image_path']
    except KeyError:
        return 0

def doSort():
    dir = "../data/COD20K/"
    train = 'train_boxes.json'
    val = 'val_boxes.json'
    test = 'test_boxes.json'

    for i in [train, val, test]:
        with open(dir + i, 'r') as reader:
            read = json.load(reader)
            read.sort(key=sort_img, reverse=False)

            with open(dir + i.split('_')[0] + "_boxes_sorted.json", 'w') as outfile:
                json.dump(read, outfile, indent=2)
                outfile.write('\n')
def cnt(file):
    tmp = {}

    with open(file, "rw") as reader:
        read = json.load(reader)
        read.sort(key=sort_img, reverse=False)
        for img in read:
            num = img["image_path"].split('/')[1].split('_')[0]
            if num in tmp:
                tmp[num] += 1
            else:
                tmp[num] = 1

    print(len(tmp))
    print(tmp)

    startShowArr(tmp, file)

def createVal():
    # Randomly select ~1000 images from train set and make ist the val set
    valset = np.random.randint(19060, size=(1, 1000))
    valset = set(valset[0])
    print(len(valset))

    train = []
    train_new = []
    val = []
    filenames = []

    with open('../data/COD20K/train_boxes.json', 'r') as reader:
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

    with open('../data/COD20K/val_boxes.json', 'w') as outfile:
        json.dump(val, outfile, indent=2)
        outfile.write('\n')
    print(len(val))

    with open('../data/COD20K/train_new_boxes.json', 'w') as writer:
        json.dump(train_new, writer, indent=2)
        writer.write('\n')
    print(len(train_new))

def startShow(show):
    with open('../data/COD20K/test_boxes.json', 'r') as reader:
        data = json.load(reader)
        data.sort(key=sort_img, reverse=False)
        for i, img in enumerate(data):
            if i == show:
                img_loaded = cv2.imread('../data/COD20K/' + img['image_path'])
                print('../data/COD20K/' + img['image_path'])
                showBB(img_loaded, img['rects'], img['image_path'])

def startShowArr(arr, file):
    a = OrderedDict(sorted(arr.items()))
    cnt = 0

    with open(file, 'r') as reader:
        data = json.load(reader)
        data.sort(key=sort_img, reverse=False)
        for k, v in a.items():
            print("Index: %d" % cnt)
            for i, img in enumerate(data):
                if i == cnt:
                    img_loaded = cv2.imread('../data/COD20K/' + img['image_path'])
                    print('../data/COD20K/' + img['image_path'])
                    showBB(img_loaded, img['rects'], img['image_path'])
            cnt += v

def showBB(img, rects, name):
    for bb in rects:
        print(bb)
        img = cv2.rectangle(img, (bb['x1'], bb['y1']), (bb['x2'], bb['y2']), (255, 0, 0), 1)

    fig = plt.gcf()
    fig.canvas.set_window_title(name)
    fig.set_size_inches( (fig.get_size_inches()[0]*3, fig.get_size_inches()[1]*3) )

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default=None, type=str)
    parser.add_argument('--show', default=None, type=int)
    parser.add_argument('--check', default=None, type=str)
    parser.add_argument('--sort', default=None, type=int)
    parser.add_argument('--cnt', default=None, type=str)
    args = parser.parse_args()

    if args.sort is not None:
        doSort()
    if args.cnt is not None:
        cnt(args.cnt)

    if args.phase is not None and not 'val':
        createJson(args.phase)
    elif args.phase == 'val':
        createVal()
    elif args.check is not None:
        checkAnno(args.check)

    if args.show is not None:
        startShow(args.show)


if __name__ == '__main__':
    main()
