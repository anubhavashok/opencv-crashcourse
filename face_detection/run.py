import numpy as np
import cv2


def displayImage(img):
    cv2.imshow('', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# First load the cascades
# Credit: https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load image and convert to grayscale

# Detect multi scale cascade

# Display detected bounding boxes

# Output results

# Display image
