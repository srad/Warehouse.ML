import glob
from os import path, mkdir, remove, listdir
import json
from .filters import activate, canny, harris
import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from datetime import datetime


intermediate_steps = True

TEMPLATE_WITH = 642
TEMPLATE_HEIGHT = 111


def read_json(file):
    with open(file) as json_file:
        return json.load(json_file)


def index(in_dir, outfile="data.json"):
    data = {}
    print("\nCreating file index data.json")
    print(f'Reading: {in_dir}')

    for name in get_files(in_dir):
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


def get_template(template_path):
    return [cv.imread(path.join(template_path, f), 0) for f in get_files(template_path)]


def match_templates_load(file, threshold, template_path):
    """
    Loads the image, converts to grayscale and the call match_templates.
    """
    in_img = cv.imread(file)
    if in_img is None:
        print(f'File not found: {file}')
        exit(0)
    
    img_gray = cv.cvtColor(in_img, cv.COLOR_BGR2GRAY)

    match_templates(img_gray, threshold, get_template(template_path), in_img)


def match_templates(grey_img, threshold, templates, in_img=None):
    """
    Expects that the passed in image is already a grayscale loaded img.
    """
    h, w = grey_img.shape
    out_img = np.zeros((h, w, 1), dtype="uint8")

    for template in templates:
        height, width = template.shape
        res = cv.matchTemplate(grey_img, template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):
            if in_img is not None:
                cv.rectangle(in_img, pt, (pt[0] + width, pt[1] + height), (0, 255, 0), 2)
            cv.rectangle(out_img, pt, (pt[0] + width, pt[1] + height), 255, cv.FILLED)

    if in_img is not None:
        cv.imwrite("matching_marks.png", in_img)
        cv.imshow("Template Matches", in_img)
        cv.waitKey(0)
        cv.imwrite("matching_locations.png", out_img)
        cv.imshow("Match BW", out_img)
        cv.waitKey(0)

    return out_img


# This script generates the bounding boxes from the annotation color
def box(in_dir, data, capture_cam):
    print("\nCreating masks and bounding boxes")
    print("cam:", capture_cam)

    length = len(data.keys())

    path_segments, path_masks, path_bounds, path_boxes = folders(in_dir)

    for i, uuid in enumerate(data):
        cams = list(data[uuid]['cams'].keys()) if capture_cam is None else [capture_cam]

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


def extract_feature_load(file, template_path):
    im = cv.imread(file)

    i_edge, i_corner = features(np.copy(im))
    i_match = match_templates(i_edge, 0.85, get_template(template_path), np.copy(im))
    i_shadow = find_shadows(np.copy(im))

    cv.imshow("Edges", i_edge)
    cv.waitKey(0)
    cv.imshow("Corners", i_corner)
    cv.waitKey(0)


def find_shadows(im):
    #im = cv.imread(file)
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    im_h, im_w = gray.shape
    out_im = np.zeros((im_h, im_w, 1), dtype="uint8")

    _, thresh = cv.threshold(gray, 50, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        if w < im_w and area > 100 and w > h and ((h / w) < 0.4):
            cv.rectangle(out_im, (x,y), (x+w,y+h), 255, cv.FILLED)
            #cv.rectangle(im, (x,y), (x+w,y+h), (0, 255, 0), 2)

    #cv.imshow("Shadows", im)
    #cv.waitKey(0)

    return out_im


def pipe(filename, steps, args=[]):
    img = cv.imread(filename)
    x = cv.imread(filename)
    for i, fn in enumerate(steps):
        x = fn((x, img))
        if intermediate_steps:
            cv.imwrite(f'pipe_{i}.png', x)
    return x


def get_files(dir):
    return [f for f in listdir(dir) if isfile(join(dir, f)) and (f.endswith(".jpg") or f.endswith(".png"))]


def plot(p, save_path=None):
    sns.set_theme()
    fig, ax = plt.subplots(1, 1, figsize=(15, 3), dpi=96)
    sns.heatmap(p, vmin=0, vmax=1)
    ax.set(xticklabels=[], yticklabels=[])
    if save_path is not None:
        plt.savefig(path.join(save_path, f'p_xy_{stamp()}.png'))
    plt.show()


def feature_template(in_dir, prefix):
    print("Creating probability template P_xy")

    w, h = TEMPLATE_WITH, TEMPLATE_HEIGHT
    path_features = path.join(in_dir, "features")
    path_results = path.join(in_dir, "results")

    if not path.exists(path_results):
        mkdir(path_results)

    files = [f for f in get_files(path_features) if f.startswith(prefix)]
    if len(files) == 0:
        print("No files to compute templates for")
        exit()
        
    p = np.zeros((h, w), dtype=np.double)

    l = len(files)
    for index, f in enumerate(files):
        file = path.join(in_dir, "features", f)
        p += np.array(cv.cvtColor(cv.imread(file), cv.COLOR_BGR2GRAY), dtype=np.double) / 255.0
        print(f'P_xy: {index}/{l}', end='\r')

    # Convert back to grayscale: [0..1] => [0..255]
    p = p / l
    print("Max: ", np.max(p))
    np.save(path.join(path_results, f'p_xy_{prefix}_{stamp()}.npy'), p)
    plot(p, path_results)

    # To image
    p = p * 255.0
    cv.imwrite(path.join(path_results, f'feature_p_xy_{prefix}_{stamp()}.png'), np.array(p, dtype="uint8"))


def load_plot(in_dir, name):
    path_results = path.join(in_dir, "results", name)

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

    # fig = plt.figure(figsize=(w / 5, h / 10))
    fig = plt.gcf()
    # fig.set_size_inches(w / 10, h / 10)
    ax = fig.add_subplot(111, projection='3d')

    map = plt.get_cmap('gist_heat')
    # Plot a 3D surface
    surf = ax.plot_surface(X, Y, np.flip(data, axis=0), cmap=map, rstride=10, cstride=10)
    # 2d projection of 3d surface with colors
    # cset = ax.contourf(X, Y, data, zdir='z', offset=0, cmap=plt.get_cmap('gist_heat'))
    fig.colorbar(surf, ax=ax)
    ax.set_xlim3d(0, w)
    ax.set_ylim3d(0, h)
    ax.set_zlim3d(0, 1.0)
    plt.show()


def features(im, edge=True, corner=True):
    # i = pipe(file, steps=[pool, cluster, edge, rgb, corner])
    #i = cv.Canny(i, 100, 255, apertureSize=3)
    i_corner = None
    i_edge = None

    kernel = np.ones((5, 5), np.uint8)

    if edge:
        edge = canny(im)
        i_edge = cv.dilate(edge, kernel, iterations=1)

    if corner:
        corner = harris(im)
        i_corner = cv.dilate(corner, kernel, iterations=1)

    return i_edge, i_corner


def extract_features(in_dir, template_path):
    """
    Expects segments sub folder ready to go
    :param in_dir:
    :return:
    """
    print("\n\nExtracting features from image segments F_xy")
    path_features, _ = folders_template(in_dir)
    templates = get_template(template_path)

    path_segments = path.join(in_dir, "segments")
    files = get_files(path_segments)

    size = len(files)

    for index, f in enumerate(files):
        file = path.join(path_segments, f)
        image = cv.imread(file)
        w, h = TEMPLATE_WITH, TEMPLATE_HEIGHT
        i = cv.resize(image, (w, h))
        i_copy = np.copy(i)
        i_edge, i_corner = features(i)
        i_match = match_templates(i_edge, 0.9, templates)
        i_shadow = find_shadows(i_copy)

        cv.imwrite(path.join(path_features, f'edge_{index}.png'), i_edge)
        cv.imwrite(path.join(path_features, f'corner_{index}.png'), i_corner)
        cv.imwrite(path.join(path_features, f'match_{index}.png'), i_match)
        cv.imwrite(path.join(path_features, f'shadow_{index}.png'), i_shadow)
        
        # i = cv.GaussianBlur(i, (5, 5), 0)
        #cv.imwrite(path.join(path_activated, f'edge_{index}.png'), i_edge)  # activate(i))
        #cv.imwrite(path.join(path_activated, f'corner_{index}.png'), i_corner)

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


def split_dict(d, n):
    dicts = []
    l = len(d) // n
    current = {}
    for i, key in enumerate(d):
        if i % l == 0:
            dicts.append(current)
            current = {}
        else:
            current[key] = d[key]

    return dicts


def load_pxy_compute_likelihood(a_xy, in_dir):
    # Make sure a_xy and p_xy have the same dimensions!
    p_xy = np.load(path.join(in_dir, f'p_xy_{stamp()}.npy'))

    file = cv.cvtColor(cv.imread(path.join(in_dir, "activated", a_xy)), cv.COLOR_BGR2GRAY)
    a_xy = np.array(file, dtype=np.double) / 255.0
    print(likelihood(p_xy, a_xy))


def get_json(json_path):
    if not path.exists(json_path):
        print(f'JSON file with index not found, first create with: python main.py index <input-dir>')
        exit(0)

    data = read_json(json_path)
    p = path.dirname(json_path)

    return data


def pipeline(in_dir, template_path, cam=None):
    index(in_dir)
    data = get_json(path.join(in_dir, "data.json"))
    box(in_dir, data, capture_cam=cam)
    extract_features(in_dir, template_path)
    feature_template(in_dir, "edge_")
    feature_template(in_dir, "corner_")
    feature_template(in_dir, "match_")
    feature_template(in_dir, "shadow_")
