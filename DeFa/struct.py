import tensorflow as tf
import os
import cv2 as cv
import numpy as np
import linecache
import math

# Basic function of network structure
def conv(name,x,kers,outs,s,bias,pad):
    shape = [i.value for i in x.get_shape()]
    ker = int(math.sqrt(kers))
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [ker,ker,shape[3],outs],
                            tf.float32,
                            tf.initializers.random_normal(stddev=0.02))
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,
                            tf.initializers.constant(0.))
        padd = "SAME" if pad else "VALID"
        if bias: 
            return tf.nn.conv2d(x,w,[1,s,s,1],padd)+b
        else:
            return tf.nn.conv2d(x,w,[1,s,s,1],padd)

def conv_block(name,x,outs):
    with tf.variable_scope(name):
        conv1 = conv('conv1',x,3*3,outs,2,False,True)
        return tf.nn.relu(conv1)

def res_block(name,x,outs):
    with tf.variable_scope(name):
        conv1 = conv('conv1',x,3*3,outs,1,False,True)
        conv_lea = tf.nn.leaky_relu(conv1,0.2)
        conv2 = conv('conv2',conv_lea,3*3,outs,1,False,True)
        ans = tf.nn.leaky_relu(conv2+x,0.2)
        return ans

def upscale_ps(name,x,outs):
    with tf.variable_scope(name):
        conv1 = conv('conv1',x,3*3,4*outs,1,False,True)
        conv_ins = tf.contrib.layers.instance_norm(conv1)
        conv_leaky = tf.nn.leaky_relu(conv_ins,0.1)
        return tf.depth_to_space(conv_leaky,2)

def Dense(name,x,outs,reshape):
    if reshape:
        ba,hi,wi,ch = [i.value for i in x.get_shape()]
        X = tf.reshape(x,[ba,-1])
        w_shape = int(hi*wi*ch)
    else:
        X = x
        ba,num = [i.value for i in x.get_shape()]
        w_shape = num
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [w_shape,outs],
                            tf.float32,
                            tf.initializers.random_normal(stddev=0.2))
        b = tf.get_variable('b',
                            [1,outs],
                            tf.float32,
                            tf.initializers.constant(0.))
        return tf.matmul(X,w)+b

def conv_block_d(name,x,outs):
    with tf.variable_scope(name):
        conv1 = conv('conv1',x,4*4,outs,2,False,True)
        conv_ins = tf.contrib.layers.instance_norm(conv1)
        ans = tf.nn.leaky_relu(conv_ins,alpha=0.2)
        return ans
        
def Encoder(name,x,re_use): # 输入： 64 * 64 * 3，输出： 4 * 4 * （512*4）
    with tf.variable_scope(name,reuse=re_use):
        conv1 = conv('conv1',x,5*5,64,1,False,True)
        block1 = conv_block('block1',conv1,128)
        block2 = conv_block('block2',block1,256)
        block3 = conv_block('block3',block2,512)
        block4 = conv_block('block4',block3,1024)
        fc1 = Dense('fc1',block4,1024,True)
        fc2 = Dense('fc2',fc1,4*4*1024,False)
        fc2_reshape = tf.reshape(fc2,[-1,4,4,1024])
        up1 = upscale_ps('up1',fc2_reshape,512)
        return up1

def Decoder_ps(name,x,re_use):
    with tf.variable_scope(name,reuse=re_use):
#         X = tf.depth_to_space(x,2)
        up1 = upscale_ps('up1',x,256)
        up2 = upscale_ps('up2',up1,128)
        up3 = upscale_ps('up3',up2,64)
        block1 = res_block('block1',up3,64)
        block2 = res_block('block2',block1,64)
        
        Alpha = conv('Alpha',block2,5*5,1,1,True,True)
        alpha = tf.nn.sigmoid(Alpha)
        
        Rgb = conv('Rgb',block2,5*5,3,1,True,True)
        rgb = tf.nn.tanh(Rgb)
        return rgb

def Rote_image(image,angle):
    height, width = image.shape[:2]
    image_center = (width/2, height/2)
    rotation_matrix = cv.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    rotated_width = int(height*abs_sin + width*abs_cos)+35
    rotated_height = int(height*abs_cos + width*abs_sin)+35
    rotation_matrix[0, 2] += rotated_width/2 - image_center[0]
    rotation_matrix[1, 2] += rotated_height/2 - image_center[1]
    return cv.warpAffine(image,rotation_matrix,(rotated_width, rotated_height)),height,width

def extract(image,old_h,old_w):
    h,w,_ = np.shape(image)
    for i in range(h):
        if np.any(image[i,:,:]):
            up = i
            break
    for j in range(w):
        if np.any(image[:,j,:]):
            left = j
            break
    extra = image[up:(up+old_h),left:(left+old_w),:]
    return extra

def angle_and_box(txt_dir):
    line = linecache.getline(txt_dir,1).split(' ')
    line[-1] = line[-1].split('\n')[0]
    angle = float(line[1])
    box = []
    for i in line[2:]:
        box.append(int(i))
    return line[0],angle,box

def get_final_image(raw_image_dir,angle,box,Fake):
    fake = (Fake+1)*127.5
    fake_resize = cv.resize(fake,(int(abs(box[0]-box[2])),int(abs(box[1]-box[3]))))
    raw_image = cv.imread(raw_image_dir)
    rote_image,old_h,old_w = Rote_image(raw_image,angle)
    rote_image[box[1]:box[3],box[0]:box[2],:] = fake_resize
    fan_angle = -1*(angle)
    rote_image_fan,_,_ = Rote_image(rote_image,fan_angle)
    final_image = extract(rote_image_fan,old_h,old_w)
    return final_image
    
