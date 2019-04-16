#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import math
import tensorflow as tf
import glog as log
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass
import pudb

from lanenet_model import lanenet_merge_model
from config import global_config
from data_provider import lanenet_data_processor_test

import numpy as np


CFG = global_config.cfg
# VGG_MEAN = [103.939, 116.779, 123.68]  # RGB?
VGG_MEAN = [123.68, 116.779, 103.939]  # BGR?


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='false')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=8)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir', default=None)
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()



def test_lanenet(dataset_dir, image_path, weights_path, use_gpu, image_list, batch_size, save_dir):

    """
    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    
    test_dataset = lanenet_data_processor_test.DataSet(image_path, batch_size, dataset_dir)
    input_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='input_tensor')
    imgs = tf.map_fn(test_dataset.process_img, input_tensor, dtype=tf.float32)
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet()
    binary_seg_ret, instance_seg_ret = net.test_inference(imgs, phase_tensor, 'lanenet_loss')
    initial_var = tf.global_variables()
    final_var = initial_var[:-1]
    saver = tf.train.Saver(final_var)
    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'GPU': 0})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=sess_config)
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=weights_path)
        for i in range(math.ceil(len(image_list) / batch_size)):
            print("Inferring batch {} with batch size {}".format(i, batch_size))
            paths = test_dataset.next_batch()
            instance_seg_image, existence_output = sess.run([binary_seg_ret, instance_seg_ret],
                                                            feed_dict={input_tensor: paths})
            # pudb.set_trace()
            for cnt, input_image_path in enumerate(paths):
                print("Generating output for input image {}".format(input_image_path))
                input_image = cv2.imread(input_image_path)
                input_image = cv2.resize(input_image, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT))

                all_lanes_image = np.zeros((CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH))

                parent_path = os.path.dirname(input_image_path)
                output_dir = os.path.join(parent_path, save_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                file_exist = open(os.path.join(output_dir, os.path.basename(input_image_path)[:-3] + 'exist.txt'), 'w')
                for cnt_img in range(4):
                    filename = os.path.join(output_dir, os.path.basename(input_image_path)[:-4] + '_' + str(cnt_img + 1) + '_avg.png')
                    lane_image = (instance_seg_image[cnt, :, :, cnt_img + 1] * 255).astype(int)
                    cv2.imwrite(filename, lane_image)
                    all_lanes_image = all_lanes_image + lane_image
                    if existence_output[cnt, cnt_img] > 0.5:
                        file_exist.write('1 ')
                    else:
                        file_exist.write('0 ')
                file_exist.close()

                # Make all_lanes_image into color image (3 channels)
                all_lanes_color_image = np.zeros((CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3))
                all_lanes_color_image[:, :, 2] = all_lanes_image[:, :]

                all_lanes_color_image_path = os.path.join(output_dir, os.path.basename(input_image_path)[:-3] + 'lanes.jpg')
                resized_input_image_path = os.path.join(output_dir, os.path.basename(input_image_path)[:-3] + 'resized_input.jpg')
                overlay_image_path = os.path.join(output_dir, os.path.basename(input_image_path)[:-3] + 'overlay.jpg')

                cv2.imwrite(resized_input_image_path, input_image)
                cv2.imwrite(all_lanes_color_image_path, all_lanes_color_image)
                cv2.imwrite(overlay_image_path, all_lanes_color_image + input_image)


    sess.close()
    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    save_dir = os.path.join(args.image_path, 'predicts')
    if args.save_dir is not None:
        save_dir = args.save_dir

    img_name = []
    with open(str(args.image_path), 'r') as g:
        for line in g.readlines():
            img_name.append(line.strip())

    test_lanenet(args.dataset_dir, args.image_path, args.weights_path, args.use_gpu, img_name, args.batch_size, save_dir)