import numpy as np
from spectral import *
import os
import matplotlib.pyplot as plt
import constants
import store
import tensorflow as tf

spectral.settings.envi_support_nonlowercase_params = True


def get_fewer_lines(mat, ammount):
    n_mat = []
    r, _, _ = mat.shape
    for i in range(0, r, int(r/ammount)):
        n_mat.append(mat[i, :, :])
    return np.array(n_mat)


def hsi2matrix(arr):
    if len(arr.shape) != 3:
        raise BaseException('A entrada deve possuir 3 dimensões')

    r, c, w = arr.shape
    return np.reshape(arr, (r*c, w))


def mat2hsi(mat, shape):
    return np.reshape(mat, (-1, shape[1], shape[2]))


def get_layer(hsi, layer):
    return hsi[:, :, layer]


def remove_pixels(cube, side, amount):
    cpy_cube = np.copy(cube)
    if side == 'top':
        cpy_cube[0:amount, :, :] = 0
    elif side == 'left':
        cpy_cube[:, 0:amount, :] = 0
    elif side == 'right':
        cpy_cube[:, -amount:, :] = 0
    else:
        cpy_cube[-amount:, :, :] = 0
    return cpy_cube


def remove_pixels_from_all_dir(cube, ammount_top, ammount_left, ammount_right, ammount_down):
    cpy_cube = np.copy(cube)
    if ammount_top != 0:
        cpy_cube = remove_pixels(cpy_cube, 'top', ammount_top)
    if ammount_left != 0:
        cpy_cube = remove_pixels(cpy_cube, 'left', ammount_left)
    if ammount_right != 0:
        cpy_cube = remove_pixels(cpy_cube, 'right', ammount_right)
    if ammount_down != 0:
        cpy_cube = remove_pixels(cpy_cube, 'down', ammount_down)
    return cpy_cube


def get_hsi_data(path):
    orig_name = [a for a in os.listdir(
        path) if '.hdr' in a and 'DARK' not in a and 'WHITE' not in a]
    dark_name = [a for a in os.listdir(path) if '.hdr' in a and 'DARK' in a]
    white_name = [a for a in os.listdir(path) if '.hdr' in a and 'WHITE' in a]

    I = open_image(os.path.join(path, orig_name[0]))
    W = open_image(os.path.join(path, white_name[0]))
    D = open_image(os.path.join(path, dark_name[0]))

    return (I.load(), W.load(), D.load())


def remove_blank_lines(mat):
    return mat[~np.all(mat == 0, axis=1)]


def which_cluster_to_mantain(mask1, mask2):
    plt.figure()
    plt.title("FIGURE 1")
    plt.imshow(get_layer(mask1, 10), cmap='gray')
    plt.figure()
    plt.title("FIGURE 2")
    plt.imshow(get_layer(mask2, 10), cmap='gray')
    plt.show()

    resp = int(input('Qual cluster deseja manter? (1/2)'))
    if resp != 1 and resp != 2:
        raise BaseException("Selected option not available.")

    return resp - 1


def get_file_cube_from_folder_to_train(folder, bac_index, filename='calib.pickle'):
    bacs = os.path.join(constants.SAVESTORE, folder)
    for i, bac in enumerate(os.listdir(bacs)):
        if i == bac_index:
            ind_bac_dir = os.path.join(bacs, bac)
            calib = store.load_pickle(filename, ind_bac_dir)
            return calib


def replace_zero_in_background(originalCube, maskedCube):
    cubecpy = np.copy(originalCube)
    for i in range(cubecpy.shape[0]):
        for j in range(cubecpy.shape[1]):
            if maskedCube[i, j, 0] == 0:
                cubecpy[i, j, :] = 0
    return cubecpy


def replace_median(cube):
    x, y, z = cube.shape
    for i in range(z):
        rows, cols = np.where(cube[:, :, i] == 0)
        for j in range(len(rows)):
            if rows[j] > 1 and cols[j] > 1 and rows[j] < x - 1 and cols[j] < y - 1:
                wdn = cube[rows[j]-1:rows[j]+2, cols[j]-1: cols[j]+2, i]
                r, _ = np.where(wdn == 0)
                if len(r) == 1:
                    wdn = np.where(wdn != 0)
                    cube[rows[j], cols[j], i] = np.median(wdn)
    return cube


def get_cube_by_index(path, index, filename):
    bac = get_dir_name(path, index)
    return store.load_pickle(filename, os.path.join(path, bac))


def get_dir_name(path, index):
    return os.listdir(path)[index]


def show_img_on_wave(cube, layer):
    mat = get_layer(cube, layer)
    plt.imshow(mat, cmap='gray')
    plt.show()


def remove_spectrum(x, s=-1, f=-1):
    ss, ff = 50, 210
    if s != -1:
        ss = s
    if f != -1:
        ff = f
    return x[:, ss:ff]


def normalize(data):
    return tf.keras.utils.normalize(data, 1)


def normalize_cube(cube):
    mat = hsi2matrix(cube)
    mat = normalize(mat)
    return mat2hsi(mat, cube.shape)


def remove_spectrum(x, indx):
    return x[:,indx]