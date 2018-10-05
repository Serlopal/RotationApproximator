import argparse
from keras.models import load_model
import numpy as np
from scipy.misc import imread, imresize
import cv2
import matplotlib.pyplot as plt
from main import rot_square_img, rotation_dataset
from keras.datasets import cifar10

def squareAngleLoss(target, pred):
    """
    custom loss for angle estimation. it takes into account
    that the distance between 1 and 359 is 2 degrees, and not 358
    """
    return abs(((target - pred) + 180) % 360 - 180)


if __name__ == "__main__":

    # parse locations of images
    parser = argparse.ArgumentParser()
    parser.add_argument('original', type=str, help='original image that has been rotated')
    parser.add_argument('rotated', type=str, help='rotated version of the original image')
    args = parser.parse_args()

    # load images, resize to CIFAR10 image diagonal size (45) and transform to grayscale and normalize
    original = cv2.cvtColor(imresize(imread(args.original), (45, 45)), cv2.COLOR_RGB2GRAY) / 255
    rotated = cv2.cvtColor(imresize(imread(args.rotated), (45, 45)), cv2.COLOR_RGB2GRAY) / 255

    input = np.stack([np.stack([rotated, original], axis=2)], axis=0)

    # load pretrained model
    model = load_model('model/RotationApproximator_CIFAR10.h5', custom_objects={'squareAngleLoss': squareAngleLoss})

    # compute estimated rotation
    estimated_rotation = np.squeeze(np.squeeze(model.predict(input)))
    # transform negative angles
    estimated_rotation = estimated_rotation if estimated_rotation > 0 else estimated_rotation + 360

    # print estimated rotation
    print("estimated rotation: {0} degrees".format(estimated_rotation))
    # print(y_test[0])