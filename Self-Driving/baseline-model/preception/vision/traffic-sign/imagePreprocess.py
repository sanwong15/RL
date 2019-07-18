
from skimage import exposure
import cv2
import numpy as np


def pre_processing_single_img(img):

    img_y = cv2.cvtColor(img, (cv2.COLOR_BGR2YUV))[:,:,0]
    img_y = (img_y / 255.).astype(np.float32)
    #img_y = exposure.adjust_log(img_y)
    img_y = (exposure.equalize_adapthist(img_y,) - 0.5)
    img_y = img_y.reshape(img_y.shape + (1,))

    return img_y

def pre_processing(X):
    print(X.shape)
    X_out = np.empty((X.shape[0],X.shape[1],X.shape[2],1)).astype(np.float32)
    print(X_out.shape)
    for idx, img in enumerate(X):
        X_out[idx] = pre_processing_single_img(img)
    return X_out


def save_preprocessed_data(X,y,path):
    d = {"features": X.astype('uint8'), "labels": y}
    with open(path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)