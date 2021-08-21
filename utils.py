import cv2
import numpy as np


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the size of the image you want to resize, and
    # get the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both width and height do not matter, rotate
    # original image
    if width is None and height is None:
        return image
    # calculate the ratio of width and size
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # resize image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def crop(frame, refPt):
    #refPt looks like [x1, y1, desired length, desired height]
    # coordinate definition
    return frame[int(refPt[1]):int(refPt[1]+refPt[3]),
                 int(refPt[0]):int(refPt[0]+refPt[2])]


def detectTooth(img, start_points, direction,
                margin_treshold=10, center_treshold=20,
                canny_treshold=(None, None)):

    # expand coords
    x, y, w, h = tuple(start_points)
    mw, mh = img.shape

    points = [max(0, x-5), max(0, y-15), min(mw, w+17), h+5]

    tooth_img = crop(img, points)
    lower, upper = canny_treshold
    if None in canny_treshold:
        med_val = np.median(tooth_img)
        lower = int(max(0, 0.7*med_val))
        upper = int(min(255, 1.3*med_val))

    tooth_img = cv2.Canny(tooth_img, min(lower, 100), min(upper, 160))

    # iverse image (detect objects)
    temp_img = tooth_img.copy()
    tooth_img[temp_img == 0] = 255
    tooth_img[temp_img != 0] = 0

    kernel = np.ones((13, 3), np.uint8)
    tooth_img = cv2.erode(tooth_img, kernel, iterations=2)

    kernel = np.ones((7, 1), np.uint8)
    tooth_img = cv2.dilate(tooth_img, kernel, iterations=4)

    contours, _ = cv2.findContours(
        tooth_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [[cv2.boundingRect(cnt), cv2.contourArea(cnt), cnt]
                for cnt in contours]

    filter_lbd = (lambda x: (x[0][1]+x[0][3] >= points[1]+points[3]-margin_treshold)
                  ) if direction == True else (lambda x: (x[0][1] <= points[1]+margin_treshold))
    contours = list(filter(filter_lbd, contours))

    CENTER = tooth_img.shape[1]//2

    # tooth is located in the center of image
    for cnt in contours:
        diff = CENTER - cnt[0][0]
        diff = diff if diff > 0 else float('inf')
        cnt.append(diff)

    contours.sort(key=lambda x: x[-1])

    if contours:
        m = min(contours, key=lambda x: x[-1])
        if m[-1] > center_treshold:
            # finded object is too far from center. repeat.
            return detectTooth(img, start_points,
                               direction, canny_treshold=(lower-10, upper-10))
        else:
            # calculate area of tooth
            tooth_img_new = np.zeros(tooth_img.shape)
            cv2.fillPoly(tooth_img_new, pts=[m[2]], color=(255, 255, 255))
            kernel = np.ones((3, 3), np.uint8)
            tooth_img_new = cv2.dilate(tooth_img_new, kernel, iterations=3)
            return [m[0], m[1]*1.5]
    else:
        return [points, 0]
