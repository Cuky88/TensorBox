# -*- coding: utf-8 -*-

# python getTestImagePath.py --path data/ownset/2a_image

from os import listdir
from os.path import isfile, join
import json
import argparse

def image_path(json):
    try:
        # Also convert to int since update_time will be string.  When comparing
        # strings, "10" is smaller than "2".
        return int(json['image_path'].split("_")[3][:-4])
    except KeyError:
        return 0

def createJson(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    j = 0
    total = []

    # filename:  2A.mov_frame_1.png
    for file in files:
        #frame = file.split('.')[0] + '.png'
        #print(file.split("_")[2][:-4])
        total.append({'image_path': path.split("/")[2] + '/' + file})
        j += 1

    print(j)

    path_json= path.split("/")[0] + "/" + path.split("/")[1]

    # lines.sort() is more efficient than lines = lines.sorted()
    total.sort(key=image_path, reverse=False)

    with open(path_json + '/predict_boxes.json', 'w') as outfile:
        json.dump(total, outfile, indent=2)
        outfile.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default=None, type=str)
    args = parser.parse_args()

    if args.path is not None:
        createJson(args.path)
    else:
        print("Enter --path /PATH/TO/IMAGES/")



if __name__ == '__main__':
    main()