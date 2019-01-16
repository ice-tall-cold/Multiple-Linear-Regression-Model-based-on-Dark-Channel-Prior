import os
import sys
import cv2
import math
import numpy as np

# parameters are set here
PATCH_SIZE = 15
HAZE_WEIGHT = 0.95
BRIGHTEST_PIXELS_PERCENTAGE = 0.001
IMG_START = 1
IMG_END = 500


def dark_channel(input_img, patch_size):
    b, g, r = cv2.split(input_img)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dcp = cv2.erode(dc, kernel)
    return dcp


def atm_light(img, dcp):
    [h, w] = img.shape[:2]
    img_size = h*w
    num_pixel = int(max(math.floor(img_size*BRIGHTEST_PIXELS_PERCENTAGE), 1))
    dark_channel_vec = dcp.reshape(img_size)
    img_vec = img.reshape(img_size, 3)

    indices = dark_channel_vec.argsort()
    indices = indices[img_size-num_pixel::]

# highest intensity in the input image I are selected as the atmospheric light.
    brightest_pixel = img_vec[indices]
    brightest_r = brightest_pixel[:, 0]
    brightest_g = brightest_pixel[:, 1]
    brightest_b = brightest_pixel[:, 2]
    A = np.zeros(3)
    A[0] = max(brightest_r)
    A[1] = max(brightest_g)
    A[2] = max(brightest_b)
    return A


def transmission_estimate(im, A, sz):
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind]/A[ind]

    transmission = 1 - HAZE_WEIGHT*dark_channel(im3, sz)
    return transmission


def guided_filter(im, p, r, eps):
    mean_i = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_ip = cv2.boxFilter(im*p, cv2.CV_64F, (r, r))
    cov_ip = mean_ip - mean_i*mean_p

    mean_ii = cv2.boxFilter(im*im, cv2.CV_64F, (r, r))
    var_i = mean_ii - mean_i*mean_i

    a = cov_ip/(var_i + eps)
    b = mean_p - a*mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a*im + mean_b
    return q


def transmission_refine(im, estimate_t):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = guided_filter(gray, estimate_t, r, eps)
    return t


def recover(im, t_estimate, A, t_bound=0.1):
    res = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind]-A[ind])/cv2.max(t_estimate, t_bound) + A[ind]
    return res


if __name__ == '__main__':
    # if images file SOTS is not at current directory, please change the data_dir
    image_path = os.path.join('images','outdoor', 'RTTS', 'JPEGImages')
    result_path = os.path.join('images', 'outdoor', 'RTTS', 'result')

    for fn in os.listdir(image_path):
        im_path = os.path.join(image_path, fn)
        fn_result = fn[:-4] +'.jpg'
        assert(os.path.exists(im_path)), 'Annotation: {} does not exist'.format(im_path)
        assert(os.path.exists(result_path)),'Annotation: {} does not exist'.format(result_path)
        #ref_src = cv2.imread(image_path)
        #ref_img = ref_src.astype('float64')/255
        src = cv2.imread(im_path)
        img = src.astype('float64')/255

        dark = dark_channel(img, PATCH_SIZE)
        A = atm_light(img, dark)
        t_estimated = transmission_estimate(img, A, PATCH_SIZE)
        t_refined = transmission_refine(src, t_estimated)
        recovered_img = recover(img, t_refined, A, 0.1)
        img_recovered = recovered_img.astype('float')*255

        cv2.imwrite(os.path.join(result_path, fn_result), img_recovered)
    #cv2.imwrite(os.path.join(write_dir, 'result9_DCP.jpg'), img_recovered)
    #cv2.imshow('Original Image', img)
    #cv2.imshow('Recovered Image', recovered_img)
    cv2.waitKey()
