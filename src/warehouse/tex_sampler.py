from matplotlib import pyplot as plt 
import cv2 as cv
import numpy as np
from random import choices
import uuid
from os import path, mkdir
import json, codecs

def draw_hist(hists):
    plt.clf()
    color = ('b','g','r')
    for channel, col in enumerate(color):
        plt.plot(hists[channel], color = col)


def json_write(file_path, hist):
    json.dump(hist, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':')) ### this saves the array in .json format


def hist(img):
    color = ('b','g','r')
    hists = []
    for channel, col in enumerate(color):
        h = cv.calcHist([img], [channel], None, [256], [0,256])
        hists.append(h)
    
    return hists


def flatten_hist(h):
    return np.array(list(map(lambda x: x[0], h))).tolist()


def mc_sample(file, out_dir, w, h, count=1):
    if not path.exists(file):
        print(f'File {file} not found')
        exit(0)

    im = cv.imread(file)
    if im is None:
        print("Image could not be read, something is wrong with the file")
    
    filename = path.splitext(path.basename(file))[0]
    sample_size = w * h

    # Abosolute frequencies
    hists = hist(im)

    # normalize from 0..1
    maximum = np.max([np.max(hists[0]), np.max(hists[1]), np.max(hists[2])])

    b = hists[0] / maximum
    g = hists[1] / maximum
    r = hists[2] / maximum

    json_write(f'{out_dir}/{filename}_hist.json', {'b': flatten_hist(hists[0]), 'g': flatten_hist(hists[1]), 'r': flatten_hist(hists[2])})

    # This is the common probablity distribution of the rgb color==index of orray
    h_sum = b + g + r

    plt.xlim([0,256])
    draw_hist(hists)
    #plt.plot(h_sum, 'k--')

    plt.title('Histogram of original texture')    

    plt.savefig(f'{out_dir}/{filename}_hist.png')

    tex = np.zeros((h, w, 3), np.uint8)

    print(w, h)

    rgb = range(256)
    for i in range(count):
        for y in range(h):
            for x in range(w):
                col_b = choices(population=rgb, weights=b)
                col_g = choices(population=rgb, weights=g)
                col_r = choices(population=rgb, weights=r)
                tex[y, x] = [col_b[0], col_g[0], col_r[0]]

        file = f'{out_dir}/{filename}_tex_{i}_{sample_size}.png'
        print("Writing", file)
        cv.imwrite(file, tex)
        cv.waitKey()

        hs = hist(tex)
        hs = hs / np.max(hs)

        draw_hist(hs)
        plt.plot(b, 'k--', color="b")
        plt.plot(g, 'k--', color="g")
        plt.plot(r, 'k--', color="r")
        #plt.plot(h_sum, 'k--')    

        plt.title(f'Histogram of monte carlo texture sample {w}x{h} (n={sample_size})')
        plt.savefig(f'{out_dir}/{filename}_hist_{i}_{sample_size}.png')