import numpy as np
import cv2 as cv
import uuid
import os


source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255


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


def match(img_rgb, threshold=0.5, *templates):
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    for t in templates:
        template = cv.imread(t, 0)
        height, width = template.shape[::]
        res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(img_rgb, pt, (pt[0] + width, pt[1] + height), (0, 255, 0), 2)

    return img_rgb


def hough(img):
    # image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta
    lines = cv.HoughLines(img, 1, np.pi / 180, 200, )
    img = rgb(img)
    print(len(lines))

    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img


def save(name, img):
    cv.imwrite(f'pipe/{name}', img)


def pipeline(filename):
    img = cv.imread(filename)
    # i0 = pixelate(img, 4)
    # i1 = cluster(i0)
    i2 = edge(img)
    # i0 = blur_gauss(img)
    i3 = corner2(rgb(i2))

    # save("i0.png", i0)
    # save("i1.png", i1)
    save("i2.png", i2)
    save("i3.png", i3)


# pooling, canny, harris
def pipeline2(filename, pool=4):
    img = cv.imread(filename)
    i0 = denoise(img)
    save("i0.png", i0)

    # i1 = pixelate(i0, pool)
    # save("i1.png", i1)
    # i1 = cluster(i0)

    i2 = edge(i0)
    save("i2.png", i2)

    i3 = corner2(rgb(i2))
    save("i3.png", i3)


def call_match(img, threshold=0.9):
    return match(img, threshold, "template2.png")
    # return match(img, threshold, "template_corner_left_top.png", "template_corner_right_top.png", "template_corner_right_bottom.png", "template_corner_left_bottom.png")


intermediate_steps = False


def pipe(filename, steps, args=[]):
    img = cv.imread(filename)
    x = cv.imread(filename)
    for i, fn in enumerate(steps):
        x = fn((x, img))
        if intermediate_steps:
            cv.imwrite(f'pipe_{i}.png', x)
    return x


def dist_transform(filename):
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    return cv.distanceTransform(opening, cv.DIST_L2, 3)


def denoise(img):
    return cv.fastNlMeansDenoisingColored(img, None, 20, 10, 7, 21)


def blur_gauss(imgs):
    img, _ = imgs
    return cv.GaussianBlur(img, (5, 5), 10)


def thres(filename):
    from matplotlib import pyplot as plt
    img = cv.imread(filename, 0)
    # global thresholding
    ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


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


from pathlib import Path


def purge():
    for p in Path("pipe").glob("pipe_*"):
        p.unlink()


def line_fit(img):
    ret, thresh = cv.threshold(img, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1, 2)
    img = rgb(img)
    for cnt in contours:
        rows, cols = img.shape[:2]
        [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        img = cv.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

    return img


from os import listdir
from os.path import isfile, join


def get_files(dir):
    return [f for f in listdir(dir) if isfile(join(dir, f))]


# pipeline("image3.png")
# dist = dist_transform("image2.jpg")
# save("dist.png", dist)
# pipeline("image2.jpg")
# show(match(rgb(dist)))
# show(edge("image2.jpg"))
# l0 = laplacian("image2.jpg")
# show(thres("image2.jpg"))
# smooth = cv.fastNlMeansDenoisingColored(cv.imread("image2.jpg"), None, 20, 10, 7, 21)
# cv.imwrite("smooth.png", smooth)
# pipeline("smooth.png")
# pipeline2("image4.jpg")
# pipeline("image5.jpg")
# pipe("image1.jpg", steps=[pixelate, cluster, edge, rgb, laplacian, to_grey, contours])

def start1():
    files = get_files()
    l = len(files)
    for index, f in enumerate(files):
        # i = pipe(f'data/{f}', steps=[pool, cluster, to_grey, edge, contours])
        i = pipe(f'data/{f}', steps=[pool, cluster, edge, rgb, laplacian, to_grey, contours])
        cv.imwrite(f'pipe/{f}', i)
        print(f'\r{round(index / l * 100, 2)}%')


def activate(im, r=1):
    h, w = im.shape[:2]
    activated = np.zeros((h, w, 1), dtype="uint8")

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
            mean = im[(x - r):(x + r), (y - r):(y + r)].mean()
            if mean > 0:
                activated = cv.rectangle(activated, (y - r, x - r), (y + r, x + r), 255)

    return activated


import seaborn as sns
import matplotlib.pyplot as plt


# 1. Convert grayscale to decimal: [0..255] => [0..1] for probability calculation
def feature_template(w, h):
    files = [file for file in get_files("../data/data3/features") if file.endswith("active.jpg")]
    p = np.zeros((h, w), dtype=np.double)

    l = len(files)
    for index, f in enumerate(files):
        file = f'data/data3/features/{f}'
        p += np.array(cv.cvtColor(cv.imread(file), cv.COLOR_BGR2GRAY), dtype=np.double) / 255.0
        print(f'\r{round(index / l * 100, 1)}%')

    # Convert back to grayscale: [0..1] => [0..255]
    p = p / l
    np.save("../models/p_xy.npy", p)

    # Plot
    sns.set_theme()
    fig, ax = plt.subplots(1, 1, figsize=(15, 3), dpi=96)
    sns.heatmap(p, vmin=0, vmax=1)
    ax.set(xticklabels=[], yticklabels=[])
    plt.show()

    # To image
    p = p * 255.0
    cv.imwrite("../data/feature_p_x_y.png", np.array(p, dtype="uint8"))


def extract_features_from_front():
    files = get_files("../data/data3/segments")
    size = len(files)
    for index, f in enumerate(files):
        file = f'data/data3/segments/{f}'
        i = pipe(file, steps=[pool, cluster, edge, rgb, corner])
        w, h = 642, 111
        i = cv.resize(i, (w, h))
        cv.imwrite(f'data/data3/features/{index}.jpg', i)
        i_activated = activate(i)
        cv.imwrite(f'data/data3/features/{index}_active.jpg', i_activated)
        print(f'\r{round(index / size * 100, 1)}%')


def likelihood(p_xy, a_xy):
    h, w = a_xy.shape[:2]
    estimation = np.zeros((h, w), dtype=np.double)
    sum = 0.0

    # Leave frame for radius
    for y in range(w):
        for x in range(h):
            p_log = np.log(p_xy[x, y]) if p_xy[x, y] > 0.0000000000001 else 0
            s = a_xy[x, y] * p_log + ((1 - a_xy[x, y]) * np.log(1 - p_xy[x, y]))
            estimation[x, y] = s
            sum += s

    cv.imwrite("../models/estimation.png", np.array((1.0 - estimation) * 255.0, dtype="uint8"))

    return sum


# extract_features_from_front()
# feature_template(642, 111)

# Make sure a_xy and p_xy have the same dimensions!!!!!!!!!!!!!!!!!!
p_xy = np.load("../models/p_xy.npy")

for f in ["data3/features/0_active.jpg"]:
    file = cv.cvtColor(cv.imread(f), cv.COLOR_BGR2GRAY)
    a_xy = np.array(file, dtype=np.double) / 255.0
    print(likelihood(p_xy, a_xy))
