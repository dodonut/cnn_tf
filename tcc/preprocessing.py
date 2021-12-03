import numpy as np
import math
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import helper
import constants
import os
from scipy.signal import savgol_filter


def calibration(I, W, D):
    row, column, wave = I.shape
    arr = np.copy(I)

    meanw = np.mean(W, axis=0)
    meand = np.mean(D, axis=0)

    for z in range(wave):
        if (z % 30 == 0):
            print('CAMADAS {}-{}'.format(z, 256 if z+30 > 256 else z+30))
        for x in range(row):
            for y in range(column):
                w = meanw[0, y, z]
                d = meand[0, y, z]
                s = I[x, y, z]

                den = w-d
                num = s-d
                if den and num/den > 0:
                    arr[x, y, z] = -math.log10(num / den)
                else:
                    arr[x, y, z] = 0
    return arr


def pca_95(x):
    scaled_data = preprocessing.scale(x)

    return PCA(n_components=0.95).fit_transform(scaled_data)


def get_clusters(x):
    pca_data = pca_95(x)
    km = KMeans(n_clusters=2).fit(pca_data)
    return km


def savitzky_golay_filter(y, window_size, order, deriv=0, rate=1):
    order_range = range(order+1)
    half_window = (window_size - 1) // 2
    b = np.mat([[k**i for i in order_range]
               for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window+1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def snv_filter(mat):
    nmat = np.copy(mat)
    mean = np.mean(mat, axis=1)
    std = np.std(mat, axis=1)
    for i in range(mat.shape[0]):
        nmat[i] = (nmat[i] - mean[i])/std[i]
    return nmat


def apply_mask(km, mat):
    mask1 = np.copy(mat)
    mask2 = np.copy(mat)
    lab = km.labels_
    for i in range(mat.shape[0]):
        if lab[i] == 0:
            mask1[i, :] = 0
        else:
            mask2[i, :] = 0

    return (mask1, mask2)


def apply_filters(mat):
    mat_cpy = np.copy(mat)
    for i in range(mat.shape[0]):
        mat_cpy[i] = savgol_filter(mat_cpy[i], 21, 2, 1)

    return snv_filter(mat_cpy)


def hsi_remove_background(mat):
    mat_cpy = apply_filters(mat)
    km = get_clusters(mat_cpy)
    m1, m2 = apply_mask(km, mat)
    return (m1, m2, mat_cpy)


def preprocess_training_data_full(choose_bac: int, semipath: str):
    """
        choose_bac is the bacteria to process (since takes forever to do all at once)
        returns a calibrated array based on dark and white hdr's, the pixels containing the bacteria (with no background) and the label for that bacteria
    """

    bac_dirs = os.listdir(constants.ORIGINAL_BAC_STORE)

    for ind, bac in enumerate(bac_dirs):
        if (choose_bac == ind):

            individual_bac_dir = os.path.join(
                os.path.join(constants.ORIGINAL_BAC_STORE, bac), semipath)

            I, W, D = helper.get_hsi_data(individual_bac_dir)

            W = helper.get_fewer_lines(W, 25)
            D = helper.get_fewer_lines(D, 25)

            arr_calib = calibration(I, W, D)

            cube = preprocess_training_data_from_calibration(arr_calib)
            return [arr_calib, cube]


def preprocess_training_data_from_calibration(arr_calib):
    cube = helper.replace_median(arr_calib)

    mat = helper.hsi2matrix(cube)

    mask1, mask2, _ = hsi_remove_background(mat)
    mask1 = helper.mat2hsi(mask1, arr_calib.shape)
    mask2 = helper.mat2hsi(mask2, arr_calib.shape)

    cluster = helper.which_cluster_to_mantain(mask1, mask2)
    retCube = mask1
    if cluster == 1:
        retCube = mask2

    return retCube[:, :, 1:256-14]


def preprocess_training_data_from_calibration_with_filters(arr_calib):
    cube = helper.replace_median(arr_calib)

    mat = helper.hsi2matrix(cube)

    mask1, mask2, mat_with_filters = hsi_remove_background(mat)
    mask1 = helper.mat2hsi(mask1, arr_calib.shape)
    mask2 = helper.mat2hsi(mask2, arr_calib.shape)
    cube_with_filters = helper.mat2hsi(mat_with_filters, arr_calib.shape)

    cluster = helper.which_cluster_to_mantain(mask1, mask2)
    retCube = cube_with_filters
    if cluster == 0:
        retCube = helper.replace_zero_in_background(retCube, mask1)
    else:
        retCube = helper.replace_zero_in_background(retCube, mask2)

    return retCube[:, :, 1:256-14]


def remove_mean_of_spectre(mat):
    return mat - np.mean(mat)
