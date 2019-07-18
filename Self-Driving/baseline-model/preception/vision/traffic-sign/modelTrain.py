import tensorflow as tf
from LeNetArch import LeNet
from sklearn.utils import shuffle
from dataAugmentation import augmentation, motion_blue
from imagePreprocess import pre_processing_single_img, pre_processing


def setTrainGraph(train_X, train_y, apply_dropout = True):
    # Create placeholders
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)
    apply_dropout = tf.placeholder(tf.bool)

    rate = 0.001
    mu = 0
    sigma = 0.1
    beta = 0.001


    weights = [
        tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 12), mean = mu, stddev = sigma)),
        tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 24), mean = mu, stddev = sigma)),
        tf.Variable(tf.truncated_normal(shape=(1188, 320), mean = mu, stddev = sigma)),
        tf.Variable(tf.truncated_normal(shape=(320, n_classes), mean = mu, stddev = sigma))
    ]
    biases = [
       tf.Variable(tf.zeros(12)),
       tf.Variable(tf.zeros(24)),
       tf.Variable(tf.zeros(320)),
       tf.Variable(tf.zeros(n_classes))
    ]


    logits = LeNet(x, weights, biases, apply_dropout)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)

    regularizer = tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
    loss = tf.reduce_mean(loss_operation + beta * regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss)

    return training_operation


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, apply_dropout: False})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def preprocessDataAugmentationCombine(X_data, y_data):
    X_data_p = pre_processing(X_data)
    y_data_p = y_data


    X_data_p_aug, y_data_p_aug = augmentation(X_data_p, y_data_p)
    X_data_p_mb, y_data_p_mb = motion_blur(X_data_p_aug, y_data_p_aug, 4)

    X_train_p = np.concatenate((X_train_p, X_train_aug_p, X_train_aug_mb_p), axis=0)
    y_train_p = np.concatenate((y_train_p, y_train_aug_p, y_train_aug_mb_p), axis=0)

    return




def train(X_train, y_train):
    # Shuffle Data
    X_train, y_train = shuffle(X_train, y_train)

    # Init Var
    global best_validation_accuracy
    best_validation_accuracy = 0.0


    # Run Model Parameters
    EPOCHS = 10
    BATCH_SIZE = 128


    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, tf.train.latest_checkpoint('.'))
        # saver.restore(sess, './lenet11')

        num_examples = len(X_train_p)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train_p, y_train_p = shuffle(X_train_p, y_train_p)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train_p[offset:end], y_train_p[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, apply_dropout: True})

            validation_accuracy = evaluate(X_valid_p, y_valid_p)
            training_accuracy = evaluate(X_train_p, y_train_p)
            print("EPOCH {} ...".format(i + 1))
            print("Training Accuracy = {:.3f}".format(training_accuracy))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
            if (validation_accuracy > best_validation_accuracy):
                best_validation_accuracy = validation_accuracy
                saver.save(sess, './lenet11')
                print("Model saved")