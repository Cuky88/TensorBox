# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import json
import argparse

def createJson(file):
    train = []
    test = []
    val = []

    # 116286 instances
    data = []
    loaded = []

    with open(file, 'r') as reader:
        obj = json.load(reader)
        loaded = obj['samples']

    for ins in loaded:
        for i in ins['instances']:
            data.append({"image_path": "images/" + i['path'], "rects": [{"x1": i['2DBB'][0], "y1": i['2DBB'][1], "x2": i['2DBB'][0]+i['2DBB'][2], "y2": i['2DBB'][1]+i['2DBB'][3]}]})

    # 2/3 for train, 1/6 for test and 1/6 for val
    num_train = (2*len(data))/3
    print(num_train)
    num_test = (len(data)-num_train)/2

    for i,f in enumerate(data):
        if i <= num_train:
            train.append(f)
        elif i <= num_train + num_test:
            test.append(f)
        else:
            val.append(f)

    with open('../data/BoxCars/train_boxes.json', 'w') as outfile:
        json.dump(train, outfile, indent=2)
        outfile.write('\n')

    with open('../data/BoxCars/test_boxes.json', 'w') as outfile:
        json.dump(test, outfile, indent=2)
        outfile.write('\n')

    with open('../data/BoxCars/val_boxes.json', 'w') as outfile:
        json.dump(val, outfile, indent=2)
        outfile.write('\n')

    print(len(train))
    print(len(test))
    print(len(val))

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None, type=str)
    parser.add_argument('--show', default=None, type=int)
    parser.add_argument('--check', default=None, type=str)
    args = parser.parse_args()

    if args.file is not None:
        createJson(args.file)
    elif args.check is not None:
        checkAnno(args.check)
    else:
        print("Error, --file must be set!")

    if args.show is not None:
        with open('../data/BoxCars/full_boxes.json', 'r') as reader:
            data = json.load(reader)
            for i, img in enumerate(data):
                if i == args.show:
                    img_loaded = cv2.imread('../data/BoxCars/' + img['image_path'])
                    print('../data/BoxCars/' + img['image_path'])
                    showBB(img_loaded, img['rects'], img['image_path'])


if __name__ == '__main__':
    main()