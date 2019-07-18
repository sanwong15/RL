from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import random
import cv2


# For reference: This is a great tool https://github.com/albu/albumentations

datagen = ImageDataGenerator(
        rotation_range=17,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.3,
        zoom_range=0.15,
        horizontal_flip=False,
        dim_ordering='tf',
        fill_mode='nearest')


# Usage: datagen.fit(X_test_,augment=True,seed=0)

def augmentation(X, y):
    for X_batch, y_batch in datagen.flow(X, y, batch_size = X.shape[0], shuffle=False):
        print("X_batch shape: ", X_batch.shape)
        X_aug = X_batch.astype('uint8')
        y_aug = y_batch

        # Not sure how this works (but I don't think it will work)
        #img_out_rgb = cv2.cvtColor(X_batch[0].astype('float32'), cv2.COLOR_BGR2RGB)
        #cv2.imwrite("out.jpg", img_out_rgb)

    return X_aug, y_aug

def motion_blue(X, y, kernel_size):
    print("input X shape: ", X.shape)
    X_out = np.empty((X.shape)).astype('uint8')
    print("X_out shape: ", X_out.shape)

    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size

    for idx, img in enumerate(X):
        X_out[idx] = cv2.filter2D(img, -1, kernel_motion_blur)

    return X_out, y



def save_augmented_data(X,y,path):
    d = {"features": X.astype('uint8'), "labels": y}
    with open(path, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)






