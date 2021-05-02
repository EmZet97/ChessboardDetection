import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import Helpers.ImageProcessing as helpers

def empty(i):
    pass

cap = cv.VideoCapture(0)

cv.namedWindow("Trackbars")
cv.createTrackbar("contrast1", "Trackbars", 0, 100, empty)
cv.createTrackbar("contrast2", "Trackbars", 0, 100, empty)
cv.createTrackbar("edges1", "Trackbars", 0, 500, empty)
cv.createTrackbar("edges2", "Trackbars", 0, 500, empty)

while True:
    # Read trackbars values
    c1 = cv.getTrackbarPos("contrast1", "Trackbars")
    c2 = cv.getTrackbarPos("contrast2", "Trackbars")
    e1 = cv.getTrackbarPos("edges1", "Trackbars")
    e2 = cv.getTrackbarPos("edges2", "Trackbars")

    # Capture frame-by-frame
    ret, frame = cap.read()
    grayImg = helpers.convert_to_grayscale(frame)
    contrasted = helpers.increase_contrast(grayImg, -5 + c1 / 10, c2)
    edges = helpers.detect_edges(contrasted, e1, e2)

    # Display the resulting frame windows
    cv.imshow('frame', frame)
    cv.imshow('grayImg', grayImg)
    cv.imshow('contrasted', contrasted)
    cv.imshow('edges', edges)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()