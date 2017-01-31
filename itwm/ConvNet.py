# A test implementation of a 1D convolutional net using tensorflow.
# Author: Valentin Tschannen
# More sophisticated examples can be found here:
# https://github.com/tensorflow/models/tree/master/inception/inception
# or here: 
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

    

import numpy as np
import tensorflow as tf

N_CLASSES = 9

def _conv1d(x, W, b, is_training=None, name='conv', strides=1):
    ''' Conv1D wrapper'''
    # input tensor x: [batch, in_eight=1, in_width, in_channels]
    # kernel tensor W: [filter_height=0, filter_width, in_channels, out_channels]
    x = tf.nn.conv2d(x, W, strides=[1, 1, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    if is_training is not None:
        x = _batch_norm(x, is_training)
    return tf.nn.relu(x, name=name)

def _conv1d_depthwise(x, W, b, is_training=None, name='conv_depthwise', strides=1):
    ''' '''
    x = tf.nn.depthwise_conv2d(x,W, strides=[1,1,strides,1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    if is_training is not None:
        x = _batch_norm(x, is_training)
    return tf.nn.relu(x, name=name)


def _pool1d(x, k=2, strides=2, pool_type='max'):
    '''Pool1D wrapper'''
    pool_dict = {'ksize':[1, 1, k, 1], 'strides':[1, 1, strides, 1], 'padding':'SAME'}
    if pool_type == 'max':
        return tf.nn.max_pool(x, **pool_dict)
    elif pool_type == 'average':
        return tf.nn.avg_pool(x, **pool_dict)
    

def _batch_norm(x, is_training, reuse=False):
    '''See: 
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py#L110 '''
    return tf.contrib.layers.batch_norm(x, 
                                        is_training=is_training,
                                        decay=0.9, 
                                        center=True, 
                                        scale=True, 
                                        activation_fn=None, 
                                        updates_collections=None,
                                        reuse=reuse,
                                        trainable=True,
                                        scope='batch_norm')

        
    
def _inception_module(x, is_training, dropout_conv=1, params={}, module_position=0):
    '''See https://arxiv.org/pdf/1409.4842v1.pdf'''
    
    #[batch, in_eight=1, in_width, in_channels]
    x_shape = x.get_shape().as_list()
    
    # PATH 1: gate
    with tf.variable_scope('gate') as scope:
        out_channels, strides = _get_params_inception(params, scope.name, module_position=module_position)
        
        std = 1.0 / np.sqrt(x_shape[3]*x_shape[2])
        kernel = tf.Variable(tf.truncated_normal([1, 1, x_shape[3], out_channels], stddev=std),name='kernel')       
        biases = tf.Variable(tf.zeros([out_channels]))
        x1 = _conv1d(x, kernel, biases, is_training=is_training, strides=strides)
    
    
    # PATH 2: gate + short filter conv (1x3)
    with tf.variable_scope('short_conv') as scope:
        out_channels_gate, strides_gate, out_channels_sc, strides_sc = _get_params_inception(params, 
                                                                                             scope.name,
                                                                                             module_position=module_position)
        
        # Gate
        std = 1.0 / np.sqrt(x_shape[3]*x_shape[2])
        kernel = tf.Variable(tf.truncated_normal([1, 1, x_shape[3], out_channels_gate], stddev=std),name='kernel')       
        biases = tf.Variable(tf.zeros([out_channels_gate]))
        x2_gate = _conv1d(x, kernel, biases, strides=strides_gate)
        
        # Short conv
        std = 1.0 / np.sqrt(x2_gate.get_shape().as_list()[3]*x2_gate.get_shape().as_list()[2])
        kernel = tf.Variable(tf.truncated_normal([1, 3, x2_gate.get_shape().as_list()[3], out_channels_sc], 
                                                 stddev=std),name='kernel')       
        biases = tf.Variable(tf.zeros([out_channels_sc]))
        x2 = _conv1d(x2_gate, kernel, biases, is_training=is_training, strides=strides_sc)
        
    
    # PATH 3: gate + long filter conv (1x5)
    with tf.variable_scope('long_conv') as scope:
        out_channels_gate, strides_gate, out_channels_lc, strides_lc = _get_params_inception(params, 
                                                                                             scope.name, 
                                                                                             module_position=module_position)

        
        # Gate
        std = 1.0 / np.sqrt(x_shape[3]*x_shape[2])
        kernel = tf.Variable(tf.truncated_normal([1, 1, x_shape[3], out_channels_gate], 
                                                 stddev=std),name='kernel')       
        biases = tf.Variable(tf.zeros([out_channels_gate]))
        x3_gate = _conv1d(x, kernel, biases, strides=strides_gate)
        # Long conv
        std = 1.0 / np.sqrt(x3_gate.get_shape().as_list()[3]*x3_gate.get_shape().as_list()[2])
        kernel = tf.Variable(tf.truncated_normal([1, 5, x3_gate.get_shape().as_list()[3], 
                                                  out_channels_lc], 
                                                 stddev=std),name='kernel')       
        biases = tf.Variable(tf.zeros([out_channels_lc]))
        x3 = _conv1d(x3_gate, kernel, biases, is_training=is_training, strides=strides_lc)
        
    
    # PATH 4 pooling + gate:
    with tf.variable_scope('pooling') as scope:
        k, out_channels, strides = _get_params_inception(params, scope.name, module_position=module_position)
        
        # Pooling
        x4_pool = _pool1d(x,k=k,strides=1)
        # Gate
        std = 1.0 / np.sqrt(x4_pool.get_shape().as_list()[3]*x4_pool.get_shape().as_list()[2])
        kernel = tf.Variable(tf.truncated_normal([1, 1, x4_pool.get_shape().as_list()[3], out_channels], 
                                                 stddev=std),name='kernel')       
        biases = tf.Variable(tf.zeros([out_channels]))
        x4 = _conv1d(x4_pool, kernel, biases, is_training=is_training, strides=strides)
    
    
    # CONCATENATE in depth
    with tf.variable_scope('concat') as scope:
        concat = tf.concat(3,[x1,x2,x3,x4])
        
    return concat
        
def _reshape_Conv_to_FullyConnected(conv):
    '''Reshape output of convolution layer for full connection'''
    shape = conv.get_shape().as_list()
    dim = shape[1]*shape[2]*shape[3]       
    return tf.reshape(conv, [-1, dim])


def _linear_activation(x, W, b, is_training=None, keep_prob=1, name=None):
    '''Linear layer with ReLU activation and dropout.'''
    l = tf.add(tf.matmul(x, W), b)
    if is_training is not None:
        l = _batch_norm(l,is_training)
    l = tf.nn.relu(l,name=name)
    return tf.nn.dropout(l, keep_prob)


def _get_params_inception(params, path, module_position=0):
    ''''''
    i = module_position
    if i > 2: i = -1
        
    if 'gate' in path:
        out_channels = params.get('gate_out_channels', [14,21,21][i])
        strides = params.get('gate_strides',1)
        return out_channels, strides
    
    if 'short_conv' in path:
        out_channels_gate = params.get('short_conv_out_channels_gate', [14,21,21][i])
        strides_gate = params.get('short_conv_strides_gate',1)
        out_channels_sc = params.get('short_conv_out_channels_sc', [21,28,28][i])
        strides_sc = params.get('short_conv_strides_sc',1)
        return out_channels_gate, strides_gate, out_channels_sc, strides_sc
    
    if 'long_conv' in path:
        out_channels_gate = params.get('long_conv_out_channels_gate', [14,21,21][i])
        strides_gate = params.get('long_conv_strides_gate',1)
        out_channels_lc = params.get('long_conv_out_channels_lc', [21,28,28][i])
        strides_lc = params.get('long_conv_strides_lc',1)
        return out_channels_gate, strides_gate, out_channels_lc, strides_lc
    
    if 'pooling' in path:
        k = params.get('pooling_k', 3)
        out_channels = params.get('pooling_out_channels_gate', [14,21,28][i])
        strides = params.get('pooling_strides_gate',1)
        return k, out_channels, strides
    
    
    else:
        raise ValueError('Inception path %s invalid' % path)
        
        
def inception_net(x, dropout_fc, is_training, dropout_conv=1, clip_norm=1e-1):
    ''' '''
    # DEPTHWISE CONV (NOT IN USE!!!!!!!)
    with tf.variable_scope('depthwise') as scope:
        channel_multiplier = 2
        xshape = x.get_shape().as_list()
        std = 1.0 / np.sqrt(xshape[2]*xshape[3])
        kernel = tf.Variable(tf.truncated_normal([1, 1, xshape[3], channel_multiplier], 
                                             stddev=std),name='kernel')       
        biases = tf.Variable(tf.zeros([xshape[3]*channel_multiplier]))
        xdw = _conv1d_depthwise(x, kernel, biases, is_training=is_training, strides=1)
        xdw = tf.nn.dropout(xdw, dropout_conv)
    
    
    # INCEPTION 1
    with tf.variable_scope('inception1') as scope:
        incept1 = _inception_module(x, is_training, dropout_conv=dropout_conv, module_position=0)
        
        
    # POOLING 1
    pool1 = _pool1d(incept1, k=2, strides=2)
    pool1 = tf.nn.dropout(pool1, dropout_conv)
    
    # INCEPTION 2
    with tf.variable_scope('inception2') as scope:
        incept2 = _inception_module(pool1, is_training,dropout_conv=dropout_conv, module_position=1)
        incept2 = tf.nn.dropout(incept2, dropout_conv)
     
    
    # INCEPTION 3
    with tf.variable_scope('inception3') as scope:
        incept3 = _inception_module(incept2, is_training, dropout_conv=dropout_conv, module_position=2)
        
                
    # POOLING 3
    pool3 = _pool1d(incept3, k=2, strides=2)
    pool3 = tf.nn.dropout(pool3, dropout_conv)
       
    # FLATTEN
    flatten = _reshape_Conv_to_FullyConnected(pool3)
    
    
    # FULLY CONNECTED 1
    with tf.variable_scope('fc1') as scope:
        n1 = 140 #112
        # Weighting and activation (with dropout)
        dim = flatten.get_shape()[1].value
        #print('Number of inputs to 1st fully connected layer: %d' % dim)
        weights = tf.Variable(tf.truncated_normal([dim,n1], stddev=1e-3))
        weights_rn = tf.clip_by_norm(weights, clip_norm, axes=[0,1], name='clip1')
        biases = tf.Variable(tf.fill([n1],0.1))
        fc1 = _linear_activation(flatten, weights_rn, biases, is_training=is_training, keep_prob=dropout_fc, name=scope.name)

        
    # FULLY CONNECTED 2
    with tf.variable_scope('fc2') as scope:
        n2 = 70 #56     
        weights = tf.Variable(tf.truncated_normal([n1,n2], stddev=1e-3))
        weights_rn = tf.clip_by_norm(weights, clip_norm, axes=[0,1], name='clip2')
        biases = tf.Variable(tf.fill([n2],0.1))
        fc2 = _linear_activation(fc1, weights_rn, biases, is_training=is_training, keep_prob=dropout_fc, name=scope.name)

                   
    # OUTPUT 
    with tf.variable_scope('output') as scope:
        weights = tf.Variable(tf.truncated_normal([n2,N_CLASSES],stddev=1e-3))
        biases = tf.Variable(tf.fill([N_CLASSES],0.1))
        output = tf.add(tf.matmul(fc2, weights), biases)
        
    return output

#norm = tf.sqrt(tf.reduce_sum(tf.square(weights), 1)
#weights_renormed = weights * tf.expand_dims(clip_norm / tf.maximum(clip_norm, norms), 1) 


