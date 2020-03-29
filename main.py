from imutils import paths
import argparse
import cv2
from random import shuffle
from image_filter import variance_of_laplacian
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# The dataset is taken from: "https://www.kaggle.com/kwentar/usage-example-of-blur-dataset"
BLUR_DATASET = r"D:\Projects\PyCharmProjects\watsonTraining\images_google\blur_dataset\blur_dataset"
FILTER_FUNC = variance_of_laplacian


def filter_image(image_path):
    """
    Gets a path of an image and outputs the blurry value of the given image.
    :param image_path: the path of an input image.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = FILTER_FUNC(gray)

    if fm > args["threshold"]:
        print(image_path + " - Not Blurry: " + str(fm))

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < args["threshold"]:
        print(image_path + " - Blurry: " + str(fm))

    img = mpimg.imread(image_path)
    imgplot = plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", default=BLUR_DATASET,
                    help="path to input directory of images")
    ap.add_argument("-t", "--threshold", type=float, default=100.0,
                    help="focus measures that fall below this value will be considered 'blurry'")
    args = vars(ap.parse_args())

    image_paths = list(paths.list_images(args["images"]))
    shuffle(image_paths)

    # loop over the input images
    for image_ath in image_paths:
        filter_image(image_ath)
