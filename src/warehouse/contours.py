import sys
import cv2 as cv
import numpy as np
from os import path, mkdir, remove, listdir
import json

def read_json(file):
    with open(file) as json_file:
        return json.load(json_file)

def contours(img):
    edges = cv.Canny(img, 100, 200, apertureSize=7)
    height, width = img.shape
    mask = np.zeros((height, width), np.uint8)

    contours, hierarchy = cv.findContours(edges, cv.RETR_CCOMP, 4)

    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        p0, p1, p2, p3 = box
        mask = cv.drawContours(mask, [box], 0, 255)

    print(len(contours))

    return mask

filename = sys.argv[1]
json = read_json(f'data/{filename}.json')

im = cv.imread(f'data/{filename}.png')
height, width, _ = im.shape

# convert the y coordinates, in unity (0, 0) screen coordinates start top left
# in opencv they start left bottom
for v in json['vertices']:
    pos = v['positions']
    start_x = int(pos[0]['x'])
    end_x = int(max(pos[1]['x'], pos[3]['x']))
    y_min = int(min(height - pos[0]['y'], height - pos[1]['y']))
    y_max = int(max(height- pos[2]['y'], height - pos[3]['y']))

    start_point = (start_x, y_min) 
    end_point = (end_x, y_max) 
    color = (0, 255, 0) 
    thickness = 2
    im = cv.rectangle(im, start_point, end_point, color, thickness) 

cv.imwrite(f'data/{filename}_debug.png', im)