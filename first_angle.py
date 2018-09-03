from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys,math,cv2
import os
import argparse
import tensorflow as tf
import numpy as np
import random
import align.detect_face
from time import sleep


def main(args):
    with tf.Graph().as_default(): 
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)    

    success_aligned = 0
    minsize = 20
    threshold = [ 0.6, 0.7, 0.7 ]
    factor = 0.709
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = args.output_dir+'/'+'bounding_boxes.txt'
    
    with open(bounding_boxes_filename, "w") as text_file:
        for cls_file in os.listdir(args.input_dir):
            for im_name in os.listdir(args.input_dir+'/'+cls_file):
                im_path = args.input_dir+'/'+cls_file+'/'+im_name
                print(im_path)
                img = misc.imread(im_path)
                img = img[:,:,0:3]
                _, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                x1,y1,x2,y2 = points[0,0],points[5,0],points[1,0],points[6,0]
                angle = 180*(math.atan((y2-y1)/(x2-x1))/math.pi)
                text_file.write('%s %f\n' % (im_path, angle))
                success_aligned += 1
    print('Number of successfully aligned images: %d' % success_aligned)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Directory with unaligned images.', default="./data/facenet/src/faceims")
    parser.add_argument('--output_dir', type=str, help='Directory with aligned face thumbnails.', default="./data/facenet/src/facesalign")
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=100)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=30)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
