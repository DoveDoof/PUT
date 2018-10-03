import tensorflow as tf
import numpy as np

def create_network(filter_shape = [3, 3], filter_depth = [10], dense_width = [450, 200]):
    # number_of_CNNlayers: amount of cnn layers, must be > 0 (int)
    # filter_size: filter used in the CNN layers ([x,y])
    # filter_depth: amount of filters in a CNN layer
    # dense_width: amount of nodes in a dense layer (list)

    variables = []

    # Create a placeholder for the minibatch input data
    x1 = tf.placeholder(tf.float32,[None, 90])

    # Shape x into the correct dimensions (9+1 fields of 3x3)
    #x_shaped = tf.reshape(x1,[-1, 10, 9, 1]) #Change this to 3x3x9 or 10 for the 9(10) boards conv
    x_shaped = tf.reshape(x1, [-1, 3, 3, 10])

    layer, w1_cnn, b1_cnn, beta, scale = create_new_conv_layer(x_shaped, 10, filter_depth[0], filter_shape, [1, 1], name='layer1')
    variables.append(w1_cnn)
    variables.append(b1_cnn)
    variables.append(beta)
    variables.append(scale)

    for i in range (2, len(filter_depth) + 1):
        layer, weights, biases, beta, scale = create_new_conv_layer(layer, filter_depth[i-2], filter_depth[i-1], filter_shape, [1, 1], name='layer' + str(i))
        variables.append(weights)
        variables.append(biases)
        variables.append(beta)
        variables.append(scale)
        # To be sure the weights and biases are cleared for the next layer:
        weights = []
        biases = []
    """
    # Now do 1x1x2 convolution to put the layers onto each other:
    layer, weights, biases, beta, scale = create_new_conv_layer(layer, filter_depth, 2, [1,1], [1, 1],
                                                   name='layer1x1x2')
    variables.append(weights)
    variables.append(biases)
    variables.append(beta)
    variables.append(scale)
    # Another 1x1 filter to create depth 1
    layer, weights, biases, beta, scale = create_new_conv_layer(layer, 2, 1, [1, 1], [1, 1],
                                                  name='layer1x1x1')
    variables.append(weights)
    variables.append(biases)
    variables.append(beta)
    variables.append(scale)
    """
    # Flatten out and add dense layers
    layer = tf.reshape(layer, [-1, 3*3 * filter_depth[-1]])

    # Create variables for the weights and biases for the fully connected layers
    # ([number of nodes from previous layer, desired nodes this layer], std)
    for dense_width in dense_width:
        last_layer_nodes = int(layer.get_shape()[-1])
        weights = tf.Variable(tf.truncated_normal((last_layer_nodes, dense_width), stddev=1. / np.sqrt(last_layer_nodes)), name='wd'+ str(i))
        biases = tf.Variable(tf.truncated_normal([dense_width], stddev=0.01), name='bd' + str(i))
        layer = tf.matmul(layer, weights) + biases
        layer = tf.nn.relu(layer)
        variables.append(weights)
        variables.append(biases)

    # The last layer must have 81 output nodes and use softmax:
    weights = tf.Variable(tf.truncated_normal((3 * 3 * filter_depth[-1] if not dense_width else dense_width, 81),
                                              stddev=1. / np.sqrt(int(layer.get_shape()[-1]))), name='w_out' + str(i))
    biases = tf.Variable(tf.truncated_normal([81], stddev=0.01), name='b_out' + str(i))
    layer = tf.matmul(layer, weights) + biases
    layer = tf.nn.relu(layer)
    variables.append(weights)
    variables.append(biases)
    output_layer = tf.nn.softmax(layer)

    # Returns the input data placeholder, output layer and all the weights (+ biases)
    return x1, output_layer, variables

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=1. / (filter_shape[0]*filter_shape[1])), name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

    # setup the convolutional layer operation
    conv_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    conv_layer += bias

    # Apply batch normalization
    batch_mean, batch_var = tf.nn.moments(conv_layer, [0])
    scale = tf.Variable(tf.ones(conv_layer.get_shape()[2:]))
    beta = tf.Variable(tf.zeros(conv_layer.get_shape()[2:]))
    batch_layer = tf.nn.batch_normalization(conv_layer, batch_mean, batch_var, beta, scale,  1e-5)

    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(batch_layer)

    """"
    # now perform max pooling
    # ksize is the argument which defines the size of the max pooling window (i.e. the area over which the maximum is
    # calculated).  It must be 4D to match the convolution - in this case, for each image we want to use a 2 x 2 area
    # applied to each channel
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    # strides defines how the max pooling area moves through the image - a stride of 2 in the x direction will lead to
    # max pooling areas starting at x=0, x=2, x=4 etc. through your image.  If the stride is 1, we will get max pooling
    # overlapping previous max pooling areas (and no reduction in the number of parameters).  In this case, we want
    # to do strides of 2 in the x and y directions.
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')
    """
    return out_layer, weights, bias, beta, scale
