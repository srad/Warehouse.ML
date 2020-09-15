import glob
import numpy as np
import cv2 as cv


#
# This script generates the bounding boxes from the annotation color
#

files = np.unique(np.array([file.split("_")[0].split("\\")[1] for file in glob.glob("data/data3/*.jpg")]))
length = len(files)

for i, file in enumerate(files):
    f0 = f'data3/{file}_0_*'
    f1 = f'data3/{file}_1_*'

    print(file)
    original = glob.glob(f0)[0]
    annotation = glob.glob(f1)[0]

    im = cv.imread(original)
    label = cv.imread(annotation)

    h, w, _ = label.shape
    h0, w0, _ = label.shape

    # loop over the image, pixel by pixel
    # for y in range(0, h):
    #    for x in range(0, w):
    #        r, g, b = label[y, x]
    #        if b == 255 and g == 1 and r == 139:
    #            label[y, x] = [255, 255, 255]
    #        else:
    #            label[y, x] = [0, 0, 0]

    # Find contours
    im_gray = cv.cvtColor(label, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(im_gray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x0, y0, w, h = cv.boundingRect(c)
        if w > 50 and h > 20:
            h += 10
            x0 -= 5
            w += 10
            y0 -= 5
            x1 = x0 + w

            # Don't overflow
            if x1 > w0:
                x1 = w0

            y1 = y0 + h

            if y1 > h0:
                y = h0

        cv.imwrite(f'data/data3/segments/{file}.jpg', im[y0:y1, x0:x1])
        cv.imwrite(f'data/data3/bounds/{file}_{x0}_{y0}_{x1}_{y1}.jpg', im)
        cv.imwrite(f'data/data3/bounds/{file}_debug_{x0}_{y0}_{x1}_{y1}.jpg', cv.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2))

    print(f'{i}/{length}')
