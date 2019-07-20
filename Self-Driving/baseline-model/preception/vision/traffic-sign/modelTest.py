import os
import numpy as np
from imagePreprocess import pre_processing

def test(path_to_signs):
    labels = {}
    X_new = list()
    y_new = np.array([], dtype=np.int)
    #path_to_signs = './new_images'
    for imagePath in os.listdir(path_to_signs):
        print(imagePath)
        image = cv2.imread(os.path.join(path_to_signs, imagePath), cv2.COLOR_BGRA2RGB)
        path = imagePath.split('_')
        image = cv2.resize(image, (32, 32))
        X_new.append(image)
        y_new = np.append(y_new, [np.int(path[0])])

    print(len(X_new), ' images of shape: ', X_new[0].shape)


def performanceEvaulation(X_new, y_new):
    X_new_p = pre_processing(X_new)

    with tf.Session() as sess:
        saver.restore(sess, './lenet11')

        test_accuracy = evaluate(X_new_p, y_new)
        print("Test Accuracy = {:.3f}".format(test_accuracy))