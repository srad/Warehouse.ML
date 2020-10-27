import numpy as np
import cv2 as cv
import uuid


def pool(imgs, size=2):
    input, _ = imgs
    # Get input size
    height, width = input.shape[:2]

    # Desired "pixelated" size
    w, h = (int(width / size), int(height / size))

    # Resize input to "pixelated" size
    temp = cv.resize(input, (w, h), interpolation=cv.INTER_LINEAR)

    # Initialize output image
    return cv.resize(temp, (width, height), interpolation=cv.INTER_NEAREST)


def cluster(imgs, K=8):
    img, _ = imgs
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


def edge(imgs):
    img, _ = imgs
    return cv.Canny(img, 100, 255, apertureSize=3)


def corner(imgs):
    img, _ = imgs
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    return cv.cornerHarris(gray, 2, 3, 0.05)


def corner2(imgs):
    img, _ = imgs
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(img, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img


def to_grey(imgs):
    img, _ = imgs
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def rgb(imgs):
    img, _ = imgs
    return cv.cvtColor(img, cv.COLOR_GRAY2RGB)


def show(img):
    cv.imshow(str(uuid.uuid4()), img)
    cv.waitKey(0)


def denoise(img):
    return cv.fastNlMeansDenoisingColored(img, None, 20, 10, 7, 21)


def blur_gauss(imgs):
    img, _ = imgs
    return cv.GaussianBlur(img, (5, 5), 10)


def edge2(filename):
    img = cv.imread(filename, 0)
    laplacian = cv.Laplacian(img, cv.CV_64F)
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    return laplacian


def laplacian(imgs):
    img, _ = imgs
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv.filter2D(img, cv.CV_32F, kernel)
    sharp = np.float32(img)
    imgResult = sharp - imgLaplacian
    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)
    return imgLaplacian


def segment(filename):
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    return sure_fg


def contours(imgs):
    img, original = imgs

    height, width = img.shape
    mask = np.zeros((height, width), np.uint8)

    # ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, 4)
    img = rgb((img, original))
    height, width, channels = img.shape
    # search below this coordinates
    h_max = height * 4 / 5

    # maximum box width with respect to image width (proportion)
    w_max = width * 3 / 4

    selected_boxes = []
    # final_boxes = []

    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        y_max = box[np.argsort(box[:, 1])][-1][1]
        if y_max > h_max:
            x_min = box[np.argsort(box[:, 0])][0][0]
            x_max = box[np.argsort(box[:, 0])][-1][0]
            y_min = box[np.argsort(box[:, 1])][0][1]
            dx = x_max - x_min
            if dx < w_max:
                dy = y_max - y_min
                if dx > 0:
                    d = dy / dx
                    area = dx * dy
                    mask = cv.drawContours(mask, [box], 0, 255, cv.FILLED)
                    masked = cv.bitwise_and(original, original, mask=mask)
                    mean = cv.mean(masked)
                    l = cv.sqrt(mean[0] ** 2 + mean[1] ** 2 + mean[2] ** 2)[0]

                    if 0.05 < d < 0.3 and 1200 < area and l < 50:
                        selected_boxes.append(box)

    # n²: relation between boxes
    # for b0, b1 in [(x, y) for x in box for y in box]:

    return img

    img = original
    for box in selected_boxes:
        img = cv.drawContours(img, [box], 0, (0, 255, 0), 2)

    return img


def activate(im, r=1):
    h, w = im.shape[:2]
    activated = np.zeros((h, w, 1), dtype="uint8")
    #org = np.copy(im)

    # Leave frame for radius
    for y in range(r, w - r):
        for x in range(r, h - r):
            # Test if in the radius something is > 0
            # +-----+----+-----+
            # |  0  |  1  | 2  |
            # +-----+-----+----+
            # |  3  |(x,y)| 4  |
            # +-----+-----+----+
            # |  5  |  6  | 7  |
            # +-----+-----+----+
            # Leads to generally lower results and more details
            # mean = im[(x - r):(x + r), (y - r):(y + r)].mean()
            # if mean > 0:
            if np.any(im[(x - r):(x + r), (y - r):(y + r)]):
                activated = cv.rectangle(activated, (y - r, x - r), (y + r, x + r), 255)

    return activated


def hist_color_img(img):
    """Calculates the histogram from a three-channel image"""

    histr = []
    histr.append(cv.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv.calcHist([img], [2], None, [256], [0, 256]))

    return histr


def harris(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    # To detect only sharp corners
    #dst = cv.cornerHarris(gray, blockSize=4, ksize=5, k=0.04)
    
    # Result is dilated for marking the corners
    #dst = cv.dilate(dst, None)
    
    # Threshold for an optimal value, it may vary depending on the image
    #img[dst >  0.01*dst.max()] = [255,255,255]
    #img[dst <= 0.01*dst.max()] = [0,0,0]
    #cv.imshow('Harris Corners(only sharp)',img)

    h, w, _ = img.shape
    output = np.zeros((h,w,1), np.uint8)

    # to detect soft corners
    dst = cv.cornerHarris(gray, blockSize=2, ksize=5, k=0.1)
    dst = cv.dilate(dst, None)
    output[dst > 0.001*dst.max()] = 255
    output[dst <= 0.001*dst.max()] = 0

    return output


def canny(img):
    edges = cv.Canny(img,400,500)
    dst = cv.dilate(edges, None)
    
    return dst
