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

def get_lot(Line):
    Dir = Line.split(' ')[0]
    file = Dir.split('/')[-2]
    pic_name = Dir.split('/')[-1]
    return file,pic_name

def rote_image(image,angle):
    height, width = image.shape[:2]
    image_center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    rotated_width = int(height*abs_sin + width*abs_cos)+35
    rotated_height = int(height*abs_cos + width*abs_sin)+35
    rotation_matrix[0, 2] += rotated_width/2 - image_center[0]
    rotation_matrix[1, 2] += rotated_height/2 - image_center[1]
    return cv2.warpAffine(image,rotation_matrix,(rotated_width, rotated_height))
    
def main(args):
    with tf.Graph().as_default(): 
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            
    minsize = 20
    threshold = [ 0.6, 0.7, 0.7 ]
    factor = 0.709
    random_key = np.random.randint(0, high=99999)
    
    bounding_boxes_filename = args.output_dir+'/'+'new_boxes.txt'
    with open(bounding_boxes_filename, "w") as text_file:
        for line in open(args.txt_dir):
            Dir,Angle = line.split(' ')
            angle = float(Angle.split('/')[0])

            img = misc.imread(Dir)
            im_rote = rote_image(img,angle)

            bounding_boxes, _ = align.detect_face.detect_face(im_rote, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces>0:
                det = bounding_boxes[:,0:4]
                det_arr = []
                img_size = np.asarray(im_rote.shape)[0:2]
                if nrof_faces>1:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det_arr.append(det[index,:])
                else:
                    det_arr.append(np.squeeze(det))
                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-args.margin/2, 0)
                    bb[1] = np.maximum(det[1]-args.margin/2, 0)
                    bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                    cropped = im_rote[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = cropped
                    f_name,Im_name = get_lot(line)
                    
                    if not os.path.exists(args.output_dir+'/'+f_name):
                        os.makedirs(args.output_dir+'/'+f_name)
                    misc.imsave(args.output_dir+'/'+f_name+'/'+Im_name,scaled)
                    text_file.write('%s %f %d %d %d %d\n' % (Dir,angle,bb[0],bb[1],bb[2],bb[3]))


    print('Number of successfully aligned images: %d' % success_aligned)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Directory with unaligned images.', default="./data/facenet/src/faceims")
    parser.add_argument('--txt_dir', type=str, help='Directory with unaligned images.', default="./data/facenet/src/facesalign/bounding_boxes.txt")
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
