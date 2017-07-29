# python predict.py --weights data/ownset/save.ckpt-110000 --predict data/ownset/2A.mov


import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imread, imresize
from scipy import misc
import cv2
from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes
import time
import argparse

def get_image_dir(args):
    weights_iteration = int(args.weights.split('-')[-1])
    expname = '_' + args.expname if args.expname else ''
    image_dir = '%s/images_%s_%d%s' % (os.path.dirname(args.weights), os.path.basename(args.predict)[:-5], weights_iteration, expname)
    return image_dir

def get_results(args, H):
    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.weights)

        pred_annolist = al.AnnoList()

        cap = cv2.VideoCapture(args.predict)
        if not cap.isOpened():
            cap.open(args.predict)

        frameNumber = 1800
        #frameNumber = 0
        cap.set(1, frameNumber)  # set frame to 0

        data_dir = os.path.dirname(args.predict)
        image_dir = get_image_dir(args)
        subprocess.call('mkdir -p %s' % image_dir, shell=True)
        i = 0
        display_iter = H['logging']['display_iter']
        start = time.time()
        while True:
            ret, frame = cap.read()
            if ret:
                dt = (time.time() - start) / (H['batch_size'] * display_iter)
                start = time.time()
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                img = imresize(frame, (H["image_height"], H["image_width"]), interp='cubic')
                feed = {x_in: img}
                (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
                pred_anno = al.Annotation()
                pred_anno.imageName = args.predict.split("/")[-1] + "_frame_" + str(int(pos_frame)) + '.png'
                new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                                use_stitching=True, rnn_len=H['rnn_len'], min_conf=H['min_conf'], tau=H['tau'], show_suppressed=H['show_suppressed'])

                for r in rects:
                    if r.x1 > r.x2:
                        r.x1, r.x2 = r.x2, r.x1
                    if r.y1 > r.y2:
                        r.y1, r.y2 = r.y2, r.y1
                    #print("x1: %s - x2: %s - y1: %s - y2: %s" % (r.x1, r.x2, r.y1, r.y2))

                pred_anno.rects = rects
                pred_anno.imagePath = os.path.abspath(data_dir)
                pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, frame.shape[0], frame.shape[1])
                pred_annolist.append(pred_anno)

                imname = '%s/%s' % (image_dir, os.path.basename(args.predict.split("/")[-1] + "_frame_" + str(int(pos_frame)) + '.png'))
                misc.imsave(imname, new_img)

                if i % 100 == 0:
                    print(str(i) + " out of " + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                    print("Time/image (ms): %.1f" % (dt * 1000))
                i += 1

            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                return pred_annolist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--expname', default='')
    parser.add_argument('--test_boxes', default=None)
    parser.add_argument('--predict', required=True)
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--show_suppressed', default=True, type=bool)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = 'hypes/%s' % args.weights.split("/")[1]+".json"
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = args.expname + '_' if args.expname else ''
    pred_boxes = '%s.%s%s' % (args.weights, expname, os.path.basename(args.predict))

    pred_annolist = get_results(args, H)
    pred_annolist.save(pred_boxes)


if __name__ == '__main__':
    main()