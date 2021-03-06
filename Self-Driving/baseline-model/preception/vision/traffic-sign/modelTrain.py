import tensorflow as tf
from LeNetArch import LeNet
from sklearn.utils import shuffle
from dataAugmentation import augmentation, motion_blue, save_augmented_data
from imagePreprocess import pre_processing_single_img, pre_processing
from sklearn import metrics
import seaborn as sn

def setTrainPipeline(X_train, one_hot_y, apply_dropout):
    # Train Pipeline: (1) Take in Tensorflow placeholder (contain training data), (2) pass it through the model (3) Calculate cross-entropy (4)
    # Expecting X_train is the Tensorflow placeholder
    # Expect the y_train already been transformed into one-hot array placeholder to calculate cross-entropy
    # Expecting apply_dropout is the Tensorflow placeholder

    # Return: training_operation (i.e: output of optimizer minimization

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


    logits = LeNet(X_train, weights, biases, apply_dropout)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)

    regularizer = tf.reduce_sum([tf.nn.l2_loss(w) for w in weights])
    loss = tf.reduce_mean(loss_operation + beta * regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss)

    return logits, cross_entropy, loss_operation, training_operation


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, apply_dropout: False})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def preprocessDataAugmentation(X_data, y_data):
    # 1. Augmentation
    X_data_aug, y_data_aug = augmentation(X_data, y_data)

    # 2. Save Augmentated and Motion Blur data first
    path = "./data/traffic-signs-data/train_aug.p"
    save_augmented_data(X_data_aug, y_data_aug, path)

    # 3. preprocess (normalization, grayscale)
    X_data_aug_p = pre_processing(X_data_aug)
    y_data_aug_p = y_data_aug_mb

    # 4. Save preprocess augmented motion blur data
    path = "./data/traffic-signs-data/train_aug_p.p"
    save_augmented_data(X_data_aug_p, y_data_aug_p, path)

    return X_data_aug_p, y_data_aug_p




def preprocessDataAugmentationMotionBlur(X_data, y_data):
    # 1. Augmentation
    X_data_aug, y_data_aug = augmentation(X_data, y_data)

    # 2. Motion Blur
    X_data_aug_mb, y_data_aug_mb = motion_blur(X_data_aug, y_data_aug, 4)

    # 3. Save Augmentated and Motion Blur data first
    path = "./data/traffic-signs-data/train_aug_mb.p"
    save_augmented_data(X_data_aug_mb, y_data_aug_mb, path)

    # 4. preprocess (normalization, grayscale)
    X_data_aug_mb_p = pre_processing(X_data_aug_mb)
    y_data_aug_mb_p = y_data_aug_mb

    # 5. Save preprocess augmented motion blur data
    path = "./data/traffic-signs-data/train_aug_mb_p.p"
    save_augmented_data(X_data_aug_mb_p, y_data_aug_mb_p, path)

    return X_data_aug_mb_p, y_data_aug_mb_p

def load_default_data():
    # default data path
    training_file = "./data/traffic-signs-data/train.p"
    validation_file = "./data/traffic-signs-data/valid.p"
    testing_file = "./data/traffic-signs-data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test




def train():
    # Init
    global best_validation_accuracy
    best_validation_accuracy = 0.0

    # Load the Data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_default_data()

    # preprocess original TRAIN DATA
    X_train_p = pre_processing(X_train)
    y_train_p = y_train


    # Augmentation + Preprocess on TRAIN DATA
    X_train_aug_p, y_train_aug_p = preprocessDataAugmentation(X_train, y_train)

    # Augmentation + Motion_blur + Preprocess on TRAIN DATA
    X_train_aug_mb_p, y_train_aug_mb_p = preprocessDataAugmentationMotionBlur(X_train, y_train)

    # Concate Augmentation + Motion_blur with original processed TRAIN DATA
    X_train_p = np.concatenate((X_train_p, X_train_aug_p, X_train_aug_mb_p), axis=0)
    y_train_p = np.concatenate((y_train_p, y_train_aug_p, y_train_aug_mb_p), axis=0)


    # TrainPipeline (input placeholder: x, one_hot_y, apple_dropout)
    # (1) Create placeholders
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)
    apply_dropout = tf.placeholder(tf.bool)
    # (2) Pipeline
    logits, cross_entropy, loss_operation, training_operation = setTrainPipeline(x, one_hot_y, apply_dropout)
    # training_operation will be used in the training sess

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()


    # TRAIN
    # (1) TRAINING Parameters
    EPOCHS = 10
    BATCH_SIZE = 128

    # (2) Start session for training
    with tf.Session() as sess:
        print("Training Session Starting...")
        print("Initializing Global Variables ... ")
        sess.run(tf.global_variables_initializer())
        print("Global Variables Initialization done")
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


    # Compute Accuracy

    with tf.Session() as sess:
        # saver.restore(sess, tf.train.latest_checkpoint('.'))
        saver.restore(sess, './lenet11')

        training_accuracy = evaluate(X_train_p, y_train_p)
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        validation_accuracy = evaluate(X_valid_p, y_valid_p)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))

        test_accuracy = evaluate(X_test_p, y_test_p)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

        # metrics
        y_p = tf.argmax(logits, 1)
        y_pred = sess.run(y_p, feed_dict={x: X_test_p, y: y_test_p, apply_dropout: False})

    print ("test accuracy:", test_accuracy)
    y_true = y_test_p
    print ("Precision", metrics.precision_score(y_true, y_pred, average='macro'))
    print ("Recall", metrics.recall_score(y_true, y_pred, average='micro'))
    print ("f1_score", metrics.f1_score(y_true, y_pred, average='weighted'))
    print ("Confusion_matrix")
    cm = metrics.confusion_matrix(y_true, y_pred)
    print (cm)
    # Save the confusion matrix in a csv file:
    np.savetxt("cm.csv", cm, delimiter=",")