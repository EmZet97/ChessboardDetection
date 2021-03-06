import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

import Helpers.ImageProcessing as helpers
import Helpers.ResourceLoader as resources


def empty(i):
    pass


def drawContours(img, contours):
    for cnt in contours:

        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02*peri, True)

        if len(approx) >= 4:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            cv.drawContours(img, cnt, -1, (b, g, r), 3)

    return img


cv.namedWindow("Trackbars")
cv.createTrackbar("contrast1", "Trackbars", 0, 100, empty)
cv.createTrackbar("contrast2", "Trackbars", 0, 100, empty)
cv.createTrackbar("edges1", "Trackbars", 0, 500, empty)
cv.createTrackbar("edges2", "Trackbars", 0, 500, empty)

for image in resources.load_camera_frame(0): #load source image: load_image('Resources/board1.jpg')
    # Read trackbars values
    c1 = cv.getTrackbarPos("contrast1", "Trackbars")
    c2 = cv.getTrackbarPos("contrast2", "Trackbars")
    e1 = cv.getTrackbarPos("edges1", "Trackbars")
    e2 = cv.getTrackbarPos("edges2", "Trackbars")

    # Create images variances
    grayImg = helpers.convert_to_grayscale(image)
    contrasted = helpers.increase_contrast(grayImg, -5 + c1 / 10, c2)
    blurImg = helpers.blur(contrasted)
    edgesImg = helpers.detect_edges(contrasted, e1, e2)

    # Count contours
    contours = helpers.get_contours(edgesImg)
    contoursImg = image.copy()
    contoursImg = drawContours(contoursImg, contours)

    # Display the resulting frame windows
    cv.imshow('frame', image)
    cv.imshow('grayImg', grayImg)
    cv.imshow('contrasted', contrasted)
    cv.imshow('edges', edgesImg)
    cv.imshow('blur', blurImg)
    cv.imshow('contours', contoursImg)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
#cap.release()
cv.destroyAllWindows()