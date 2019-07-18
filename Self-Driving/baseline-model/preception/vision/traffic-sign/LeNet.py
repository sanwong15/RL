from tensorflow.contrib.layers import flatten


def LeNet(x, weights, biases, apply_dropout):
    if apply_dropout is not None:
        print ("Training phase -> perform Dropout")
    else:
        print ("Evalutation phase -> not performing Dropout")

    layer = 0
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x12.
    conv1 = tf.nn.conv2d(x, weights[layer], strides=[1, 1, 1, 1], padding='VALID') + biases[layer]
    layer += 1

    # Activation.
    conv1 = tf.nn.relu(conv1, name='act1')

    # Pooling. Input = 28x28x12. Output = 14x14x12.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Dropout
    conv1 = tf.cond(apply_dropout, lambda: tf.nn.dropout(conv1, keep_prob=0.8), lambda: conv1)

    # Layer 2: Convolutional. Output = 10x10x24.
    conv2 = tf.nn.conv2d(conv1, weights[layer], strides=[1, 1, 1, 1], padding='VALID') + biases[layer]
    layer += 1

    # Activation.
    conv2 = tf.nn.relu(conv2, name='act2')

    # Pooling. Input = 10x10x24. Output = 5x5x24.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Dropout
    conv2 = tf.cond(apply_dropout, lambda: tf.nn.dropout(conv2, keep_prob=0.7), lambda: conv2)

    # Input = 14x14x12. Output = 7x7x12 = 588
    conv1_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    shape = conv1_1.get_shape().as_list()
    conv1_1 = tf.reshape(conv1_1, [-1, shape[1] * shape[2] * shape[3]])

    # Flatten conv2 Input = 5x5x24. Output = 600
    shape = conv2.get_shape().as_list()
    conv2 = tf.reshape(conv2, [-1, shape[1] * shape[2] * shape[3]])

    fc0 = tf.concat(1, [conv1_1, conv2])
    # Layer 3: Fully Connected. Input = 588+600 = 1188. Output = 320.
    fc1 = tf.matmul(fc0, weights[layer]) + biases[layer]
    layer += 1

    # Activation
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.cond(apply_dropout, lambda: tf.nn.dropout(fc1, keep_prob=0.6), lambda: fc1)
    logits = tf.matmul(fc1, weights[layer]) + biases[layer]


    # Extra Layers (Currently Disabled)
    # Layer 4: Fully Connected. Input = 512, Output = 700
    fc


    return logits

