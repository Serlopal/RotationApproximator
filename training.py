from keras.datasets import cifar100 as dataset
from scipy.misc import imread, imresize
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from random import randint
from keras.models import load_model
import keras
from sklearn.metrics import mean_absolute_error, median_absolute_error
import keras.backend as K


def rot_square_img(img, angle):
    height, width = img.shape[:2]
    half = img.shape[0]/2
    bound_d = int(np.sqrt(2)*height)

    rotation_mat = cv2.getRotationMatrix2D((half, half), angle, 1.0)
    rotation_mat[:2, 2] += bound_d/2 - half

    rotated_mat = cv2.warpAffine(img, rotation_mat, (bound_d, bound_d))

    return rotated_mat


def create_model():

    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(dim, dim, 2)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))


    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model


def rotation_dataset(imgs):

    x = []
    y = []
    img_rotations = 5

    for img in imgs:
        pad_image = rot_square_img(img, 0)

        for i in range(img_rotations):
            angle = randint(0, 359)
            rot_pad_image = rot_square_img(img, angle)
            y.append(angle)
            x.append(np.stack([cv2.cvtColor(rot_pad_image, cv2.COLOR_RGB2GRAY)/255, cv2.cvtColor(pad_image, cv2.COLOR_RGB2GRAY)/255], axis=2))

    x = np.stack(x, axis=0)
    return x, y


def absAngleLoss(target, pred):
    """
    custom loss for angle estimation. it takes into account
    that the distance between 1 and 359 is 2 degrees, and not 358
    absolute error version for model evaluation
    """
    return abs(((target - pred) + 180) % 360 - 180)


def squareAngleLoss(target, pred):
    """
    custom loss for angle estimation. it takes into account
    that the distance between 1 and 359 is 2 degrees, and not 358
    """
    return abs(((target - pred) + 180) % 360 - 180)

if __name__ == "__main__":

    # whether to train the model or used an already trained version
    train_model = True
    # whether to save the model after training
    save_model = True
    # proportion of the training set to be used as the validation set
    val_frac = 0.3
    # whether to plot the training loss
    plot_loss = True
    # whether to test the performance of the model on the test set
    test_model = True
    # number of training epochs
    num_epochs = 10

    # load image dataset
    (train_images, _), (test_images, _) = dataset.load_data()
    train_images, validation_images = train_images[:-int(len(train_images)*val_frac)], train_images[-int(len(train_images)*val_frac):]

    # create specific rotation dataset
    x_train, y_train = rotation_dataset(train_images)
    x_val, y_val = rotation_dataset(validation_images)
    x_test, y_test = rotation_dataset(test_images)

    dim = x_train.shape[1]

    if train_model:
        # train model
        model = create_model()
        model.compile(loss=squareAngleLoss, optimizer=optimizers.Adam())
        history = model.fit(x_train, y_train, batch_size=32, epochs=num_epochs,
                            validation_data=(x_val, y_val),
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)])

        if plot_loss:
            # visualizing losses and accuracy
            fig = plt.figure()
            axis = range(len(history.history['loss']))

            plt.plot(axis, history.history['loss'], label='training loss')
            plt.plot(axis, history.history['val_loss'], label='validation loss')
            plt.legend()

            plt.savefig('loss_evol.png', dpi=300)

            plt.show()


        if save_model:
            model.save('model/RotationApproximator.h5')
    else:
        model = load_model('model/RotationApproximator.h5', custom_objects={'squareAngleLoss': squareAngleLoss})

    if test_model:
        print("--- EVALUATION OF THE MODEL IN TEST DATASET ---")
        preds = np.squeeze(model.predict(x_test))

        print("Absolute error distribution description")
        angleError = absAngleLoss(np.array(y_test), preds)
        print(pd.Series(angleError).describe())

        for x in zip(np.array(y_test), preds):
            print(x)


