# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:56:02 2017

@author: Shawn
"""

import tensorflow as tf
import numpy as np

def conv2d(layer_name, in_tensor, out_channels, last_layer_ksize = 3, kernel_size = 3, stride = 1,
                 is_training = True):

    with tf.variable_scope(layer_name):
        in_channels = in_tensor.get_shape().as_list()[-1]
        w_shape = [kernel_size, kernel_size, in_channels, out_channels]
        init_stddev = np.sqrt(2/(in_channels * last_layer_ksize**2))
        w = tf.get_variable('f', dtype = tf.float32, initializer = tf.truncated_normal(
                shape = w_shape, stddev = init_stddev))
        
        h = tf.nn.conv2d(in_tensor, w, strides = [1, stride, stride, 1],
                                  padding = 'SAME')
        ##remember add the update_ops as a dependency to train_op during training
        h_norm = tf.layers.batch_normalization(inputs = h, axis = -1, training = is_training)
        out_tensor = tf.nn.relu(h_norm, name = 'relu_act')
        
        return out_tensor

def max_pool(input_tensor, kernel_size = 2, stride = 2):
    return tf.nn.max_pool(input_tensor, ksize = [1,kernel_size, kernel_size, 1],
                          strides = [1,stride,stride,1], padding = 'SAME')

def deconv2d(layer_name, in_tensor, out_channels, upsample_factor = 2, is_training = True):
    """
    the filter of the deconvoluation layers initialized with a bilinear interpolation
    filter
    """
    def bilinear_filter(filter_shape, upsample_factor):
        k_size = filter_shape[1]
        if k_size %2 == 1: 
            center_location = upsample_factor - 1
        else:
            center_location = upsample_factor - 0.5
        og = np.ogrid[:k_size, :k_size]
        bilinear_kernel = (1-np.abs(og[0] - center_location)/upsample_factor)* \
                          (1-np.abs(og[1] - center_location)/upsample_factor)
        init_w = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            for j in range(filter_shape[3]):
                init_w[:,:,i,j] = bilinear_kernel
        
        init = tf.constant_initializer(value = init_w, dtype = tf.float32)
        bilinear_weight = tf.get_variable(name = 'deconv_filter', shape = filter_shape,
                                 initializer = init)
        
        return bilinear_weight
            
    with tf.variable_scope(layer_name):
        kernel_size = 2*upsample_factor - upsample_factor%2
        stride = upsample_factor #the stride of deconv equals to the upsample_factor
        strides = [1, stride, stride, 1]
        
        out_h = in_tensor.get_shape().as_list()[1] * upsample_factor
        out_w = in_tensor.get_shape().as_list()[2] * upsample_factor
        out_shape = [in_tensor.get_shape().as_list()[0], out_h, out_w, out_channels]
        
        filter_shape = [kernel_size, kernel_size, out_channels, in_tensor.get_shape().as_list()[-1]]
        w = bilinear_filter(filter_shape, upsample_factor) #initialize w with bilinear interpolation filter
        
        deconv_tensor = tf.nn.conv2d_transpose(in_tensor, w, out_shape, strides = strides, padding = 'SAME')
        norm_deconv = tf.layers.batch_normalization(inputs = deconv_tensor, axis = -1, training = is_training)
        out_tensor = tf.nn.relu(norm_deconv, name = 'relu_act')
        
        return out_tensor

def concat_connection(tensor1, tensor2):
    return tf.concat([tensor1, tensor2], axis = -1, name = 'concat')


def soft_max(in_tensor):
    exp_tensor = tf.exp(in_tensor)
    sum_exp = tf.reduce_sum(exp_tensor, axis = 3, keep_dims = True)
    #create the a sum_exp tensor with same dimension of exp_tensor by tile along the last axis
    temp_sum_exp = tf.tile(sum_exp, [1,1,1,in_tensor.get_shape().as_list()[-1]])
    
    return tf.div(exp_tensor, temp_sum_exp)

def pixel_softmax_cost_func(in_tensor, label_tensor):
    H = in_tensor.get_shape().as_list()[1]
    W = in_tensor.get_shape().as_list()[2]
    assert H == label_tensor.get_shape().as_list()[1] & W == label_tensor.get_shape().as_list()[2]
    
    flatten_in_tensor = tf.reshape(in_tensor, [-1,H*W])
    print(flatten_in_tensor)
    flatten_label_tensor = tf.reshape(label_tensor, [-1, H*W])
    print(flatten_label_tensor)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = flatten_label_tensor,
                                                                  logits = flatten_in_tensor))
    
    return cost

#the test code
if __name__ == '__main__':
#    m = deconv2d('deconv_layer', tf.ones((2,4,4,8)), 4, 2)
#    n = conv2d('conv_layer', tf.ones((1,32,32,8)), 4)
#    n_max_pool = max_pool(n)
    
    test_out = tf.ones((1,512,512,1))
    test_label = tf.ones((1,512,512,1))
    cost = pixel_softmax_cost_func(test_out, test_label)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
#    print(sess.run(n_max_pool))
#    print(sess.run(m))
#    print(sess.run(n))
    print(sess.run(cost))
    
