import glob
from os import path, mkdir, remove, listdir
import json
from .filters import laplacian, contours, corner, corner2, cluster, denoise, edge, rgb, to_grey, pool, activate
import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import current_process


intermediate_steps = True

TEMPLATE_WITH = 642
TEMPLATE_HEIGHT = 111


def read_json(file):
    with open(file) as json_file:
        return json.load(json_file)


def index(in_dir, outfile="data.json"):
    data = {}
    p = path.join(in_dir, "*.jpg")
    print("\nCreating file index data.json")
    print(f'Reading: {p}')

    for file in glob.glob(p):
        name = file.split("\\")[1]
        parts = name.split(".jpg")[0].split("_")

        if not parts[0] in data:
            data[parts[0]] = {
                'cams': {},
                'load': parts[2] == "1"
            }

        if not parts[3] in data[parts[0]]['cams']:
            data[parts[0]]['cams'][parts[3]] = {}

        if parts[1] == "0":
            data[parts[0]]['cams'][parts[3]]['original'] = name
        if parts[1] == "1":
            data[parts[0]]['cams'][parts[3]]['mask'] = name

        # data.append({'uuid': parts[0], 'type': parts[1], 'loaded': parts[2], 'cam': parts[3]})

    # print(json.dumps(data, indent=2, sort_keys=True))

    out_path = path.join(in_dir, 'data.json')
    with open(out_path, 'w') as outfile:
        json.dump(data, outfile, indent=2, sort_keys=True)
    print(f'Written file index to: {out_path}')


def create_or_clear(paths):
    for p in paths:
        if not path.exists(p):
            mkdir(p)

        files = [f for f in listdir(p) if f.endswith(".jpg") or f.endswith(".png")]
        for f in files:
            remove(path.join(p, f))


def folders(in_dir):
    path_segments = path.join(in_dir, "segments")
    path_bounds = path.join(in_dir, "bounds")
    path_masks = path.join(in_dir, "masks")
    path_boxes = path.join(in_dir, "boxes")

    create_or_clear([path_segments, path_masks, path_bounds, path_boxes])

    return path_segments, path_masks, path_bounds, path_boxes


def folders_template(dir):
    path_features = path.join(dir, "features")
    path_activated = path.join(dir, "activated")

    create_or_clear([path_features, path_activated])

    return path_features, path_activated


def show(im):
    cv.imshow("joo", im)
    cv.waitKey(0)


# This script generates the bounding boxes from the annotation color
def box(json_path, cam=None):
    if not path.exists(json_path):
        print(f'JSON file with index not found, first create with: python main.py index <input-dir>')
        exit(0)

    print("\nCreating bounding boxes from masks")
    print("cam:", cam)

    data = read_json(json_path)
    in_dir = path.dirname(json_path)
    length = len(data.keys())

    path_segments, path_masks, path_bounds, path_boxes = folders(in_dir)

    for i, uuid in enumerate(data):
        cams = list(data[uuid]['cams'].keys()) if cam == None else [cam]

        for cam in cams:
            original = path.join(in_dir, data[uuid]['cams'][cam]['original'])
            annotation = path.join(in_dir, data[uuid]['cams'][cam]['mask'])

            # print(f'Original: {original}')
            # print(f'Annotation: {annotation}\n')

            if not (path.exists(original) or path.exists(annotation)):
                print("Not found:")
                print(f'File: {original}')
                print(f'File: {annotation}\n')
                continue

            im = cv.imread(original)
            if im is None or np.shape(im) == ():
                print(f'Error reading: {original}')
                continue

            label = cv.imread(annotation)

            if label is None or np.shape(label) == ():
                print(f'Error reading: {label}')
                continue

            h, w, _ = label.shape
            h0, w0, _ = label.shape

            lower_black = np.array([326 / 2 - 1, 0, 0], dtype="uint16")
            upper_black = np.array([326 / 2 + 1, 255, 255], dtype="uint16")

            black_mask = cv.inRange(cv.cvtColor(label, cv.COLOR_BGR2HSV), lower_black, upper_black)

            # Find contours
            # im_gray = cv.cvtColor(black_mask, cv.COLOR_BGR2GRAY)
            # ret, thresh = cv.threshold(black_mask, 254, 255, 0)
            contours, hierarchy = cv.findContours(black_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for con in contours:
                x0, y0, bw, bh = cv.boundingRect(con)
                if bw > 50 and bh > 20:
                    bh = min(h, bh + 10)
                    x0 = max(x0 - 5, 0)
                    bw = min(bw + 10, w)
                    y0 = max(y0 - 5, 0)
                    x1 = x0 + bw

                    # Don't overflow
                    if x1 > w0:
                        x1 = w0

                    y1 = y0 + bh

                    if y1 > h0:
                        y1 = h0

                    out_segments = path.join(path_segments, f'{uuid}_{cam}.png')
                    out_bounds = path.join(path_bounds, f'{uuid}_{cam}_{x0}_{y0}_{x1}_{y1}.jpg')
                    # print(path_segments, uuid, cam)
                    # out_debug = path.join(path_boxes, f'{uuid}_{c}_{x0}_{y0}_{x1}_{y1}.jpg')

                    cv.imwrite(out_segments, im[y0:y1, x0:x1])
                    cv.imwrite(out_bounds, im)
                    # print(f'Writing {out_segments}')
                    # cv.imwrite(out_debug, cv.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 2))

        print(f'{i}/{length}', end='\r')


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
        print(f'{index}/{l}', end='\r')


def plot(p, save_path=None):
    sns.set_theme()
    fig, ax = plt.subplots(1, 1, figsize=(15, 3), dpi=96)
    sns.heatmap(p, vmin=0, vmax=1)
    ax.set(xticklabels=[], yticklabels=[])
    if save_path is not None:
        plt.savefig(path.join(save_path, f'p_xy_{stamp()}.png'))
    plt.show()


def feature_template(in_dir):
    """Expects that extract_features() had been run and the "features" folder exists.
    1. Convert grayscale to decimal: [0..255] => [0..1] for probability calculation
    :param in_dir:
    :return:
    """

    print("\nCreating probability template P_xy")
    w, h = TEMPLATE_WITH, TEMPLATE_HEIGHT
    path_activated = path.join(in_dir, "activated")
    path_results = path.join(in_dir, "results")

    if not path.exists(path_results):
        mkdir(path_results)

    files = get_files(path_activated)
    p = np.zeros((h, w), dtype=np.double)

    l = len(files)
    for index, f in enumerate(files):
        file = path.join(in_dir, "activated", f)
        p += np.array(cv.cvtColor(cv.imread(file), cv.COLOR_BGR2GRAY), dtype=np.double) / 255.0
        print(f'P_xy: {index}/{l}', end='\r')

    # Convert back to grayscale: [0..1] => [0..255]
    p = p / l
    np.save(path.join(path_results, "p_xy.npy"), p)
    plot(p, path_results)

    # To image
    p = p * 255.0
    cv.imwrite(path.join(path_results, f'feature_p_x_y_{stamp()}.png'), np.array(p, dtype="uint8"))


def load_plot(in_dir):
    path_results = path.join(in_dir, "results", "p_xy.npy")

    if not path.exists(path_results):
        print(f'File not found: {path_results}')
        exit(0)

    p = np.load(path_results)
    plot_3d(p)


def plot_3d(data):
    h, w = data.shape
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(w / 5, h / 10))
    fig = plt.gcf()
    fig.set_size_inches(w / 10, h / 10)
    ax = fig.add_subplot(111, projection='3d')

    map = plt.get_cmap('gist_heat')
    # Plot a 3D surface
    surf = ax.plot_surface(X, Y, data, cmap=map, rstride=2, cstride=4, linewidth=0)
    # 2d projection of 3d surface with colors
    # cset = ax.contourf(X, Y, data, zdir='z', offset=0, cmap=plt.get_cmap('gist_heat'))
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_xlim3d(0, w)
    ax.set_ylim3d(0, h)
    ax.set_zlim3d(0, 4)
    plt.show()


def extract_features(in_dir):
    """
    Expects segments sub folder ready to go
    :param in_dir:
    :return:
    """
    print("\n\nExtracting features from image segments F_xy")
    path_features, path_activated = folders_template(in_dir)

    path_segments = path.join(in_dir, "segments")
    files = get_files(path_segments)
    size = len(files)
    for index, f in enumerate(files):
        file = path.join(path_segments, f)
        i = pipe(file, steps=[pool, cluster, edge, rgb, corner])
        w, h = TEMPLATE_WITH, TEMPLATE_HEIGHT
        i = cv.resize(i, (w, h))
        cv.imwrite(path.join(path_features, f'{index}.png'), i)
        cv.imwrite(path.join(path_activated, f'{index}.png'), activate(i))
        print(f'Feature extraction: {index}/{size}', end='\r')

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

    cv.imwrite("../../models/estimation.png", np.array((1.0 - estimation) * 255.0, dtype="uint8"))

    return sum


def stamp():
    return datetime.now().strftime("%Y_%b_%d__%H_%M_%S")


def load_pxy_compute_likelihood(a_xy, in_dir):
    # Make sure a_xy and p_xy have the same dimensions!
    p_xy = np.load(path.join(in_dir, f'p_xy_{stamp()}.npy'))

    file = cv.cvtColor(cv.imread(path.join(in_dir, "activated", a_xy)), cv.COLOR_BGR2GRAY)
    a_xy = np.array(file, dtype=np.double) / 255.0
    print(likelihood(p_xy, a_xy))


def full(in_dir, **kwargs):
    index(in_dir)
    box(path.join(in_dir, "data.json"), **kwargs)
    extract_features(in_dir)
    feature_template(in_dir)


def thread(fn, **kwargs):
    tpool = ThreadPool(4)

    # Open the URLs in their own threads
    # and return the results
    results = tpool.map(fn, **kwargs)

    # Close the pool and wait for the work to finish
    tpool.close()
    tpool.join()
