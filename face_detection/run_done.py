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
# Since we are concerned about the variations in brightness for this task
img = cv2.imread('data/img1.jpg')
displayImage(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
displayImage(gray)

# Detect multi scale cascade
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# This should now contain all faces in image

# Display detected bounding boxes
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

# Output results
cv2.imwrite('output/img1.jpg', img)

# Display image
displayImage(img)
