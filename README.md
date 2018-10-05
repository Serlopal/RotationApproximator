# RotationApproximator

CNN architecture that takes an original image and a rotated version of it, and estimates the rotation angle.


This project attempts to create a convolutional neural network architecture that can take two images as input and, given that one of them is the rotated version of the other (both padded so that the rotated image fits fully), the network will approximate the angle of rotation of the second with respect to the original.

The training script is used to train the neural network, and it's based on rotated combinations of the CIFAR10/100 public dataset. Multiple training parameters can be adjusted to tune the model's performance.

The RotationApproximator script is the final program, that takes the path to two images as input, and estimates the rotation between the two.
