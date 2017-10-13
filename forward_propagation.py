# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:25:35 2017

@author: Shawn
"""

from forward_propagation_utils import conv2d, max_pool, deconv2d, concat_connection
import tensorflow as tf
import numpy as np

def forward_propagation(X, is_training):
    ###downsample process
    d_1_1 = conv2d('down1_layer1', X, 64, is_training = is_training)
    d_1_2 = conv2d('down1_layer2', d_1_1, 64, is_training = is_training)
    d_1_3 = conv2d('down1_layer3', d_1_2, 64, is_training = is_training)
    
    d_2_1 = max_pool(d_1_3)
    d_2_2 = conv2d('down2_layer2', d_2_1, 128, is_training = is_training)
    d_2_3 = conv2d('down2_layer3', d_2_2, 128, is_training = is_training)
    
    d_3_1 = max_pool(d_2_3)
    d_3_2 = conv2d('down3_layer2', d_3_1, 256, is_training = is_training)
    d_3_3 = conv2d('down3_layer3', d_3_2, 256, is_training = is_training)
    
    d_4_1 = max_pool(d_3_3)
    d_4_2 = conv2d('down4_layer2', d_4_1, 512, is_training = is_training)
    d_4_3 = conv2d('down4_layer3', d_4_2, 512, is_training = is_training)
    
    ###the bottom layer, minimal resolution
    b_1 = max_pool(d_4_3)
    b_2 = conv2d('bottom_layer2', b_1, 1024, is_training = is_training)
    b_3 = conv2d('bottom_layer3', b_2, 1024, is_training = is_training)
    
    ####upsample process
    u_4_1 = concat_connection(d_4_3, deconv2d('up4_layer1', b_3, 512, is_training = is_training))
    u_4_2 = conv2d('up4_layer2', u_4_1, 512, is_training = is_training)
    u_4_3 = conv2d('up4_layer3', u_4_2, 512, is_training = is_training)
    
    u_3_1 = concat_connection(d_3_3, deconv2d('up3_layer1', u_4_3, 256, is_training = is_training))
    u_3_2 = conv2d('up3_layer2', u_3_1, 256, is_training = is_training)
    u_3_3 = conv2d('up3_layer3', u_3_2, 256, is_training = is_training)
    
    u_2_1 = concat_connection(d_2_3, deconv2d('up2_layer1', u_3_3, 128, is_training = is_training))
    u_2_2 = conv2d('up2_layer2', u_2_1, 128, is_training = is_training)
    u_2_3 = conv2d('up2_layer3', u_2_2, 128, is_training = is_training)
    
    u_1_1 = concat_connection(d_1_3, deconv2d('up1_layer1', u_2_3, 64, is_training = is_training))
    u_1_2 = conv2d('up1_layer2', u_1_1, 64, is_training = is_training)
    u_1_3 = conv2d('up1_layer3', u_1_2, 64, is_training = is_training)
    
    ###the output layer
    out_put = conv2d('out_layer', u_1_3, 1, kernel_size = 1, is_training = is_training)
    out_put = tf.add(out_put, X)
    
    return out_put

if __name__ == '__main__':

    X = tf.placeholder(dtype = tf.float32, shape = [1,512,512,1])
    test_output = forward_propagation(X, True)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(test_output, feed_dict = {X:np.ones((1,512,512,1))})
    print(test_output)
    
    