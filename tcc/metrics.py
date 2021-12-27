import helper
import numpy as np
import math
import matplotlib.pyplot as plt
import constants
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import os
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cnn
from matplotlib.transforms import Bbox


from constants import *

Join = os.path.join


def plot_dif_spectrum_refs(refs: list, labels: list, ismat=False, title=None, plotTest=True, onlyCurves=False, saveDir=None):
    mats = refs
    if not ismat:
        mats = []
        for i in refs:
            mats.append(helper.hsi2matrix(i))

    xmin = mats[0].shape[0]
    for i in mats:
        xmin = min(xmin, i.shape[0])

    means = []
    for i in range(len(mats)):
        mats[i] = mats[i][:xmin, :]
        # mats[i] = mats[i] - np.mean(mats[i])
        means.append(np.mean(mats[i], axis=0))

    s = ""
    if not onlyCurves:
        for i in range(0, len(mats), 2):
            s += "BAC: {}\n".format(labels[i//2])
            s += "RMSE: {}\nMean: {}\n\n".format(
                math.sqrt(np.mean(np.square(mats[i] - mats[i+1]))), np.mean(mats[i]) - np.mean(mats[i+1]))

    plt.figure(figsize=(7, 7))
    x = np.linspace(900, 2514, mats[0].shape[1])
    for i in range(len(means)):
        # plt.plot(x, means[i], '-', color=COLORS_HEX[labels[i]], linewidth=2,
        #          label='{}'.format(labels[i]))
        plt.plot(x, means[i], '-', linewidth=2,
                 label='{}'.format(labels[i]))

    plt.figlegend(bbox_to_anchor=(.8, .95), loc='upper left',
                  borderaxespad=0., fontsize=12)

    plt.xlabel("Comprimento de onda (nm)")
    plt.ylabel("Pseudo-absortância")
    if title is not None:
        plt.title(title)
    plt.show()
    if saveDir is not None:
        plt.savefig(saveDir)


def plot_spectre(cube, isCube=True):
    mat = cube
    if isCube:
        mat = helper.hsi2matrix(cube)
    nn = np.mean(mat, axis=0)
    x = np.linspace(0, mat.shape[1], mat.shape[1])
    plt.xlabel("Comprimento de onda (nm)")
    plt.ylabel("Pseudo-absortância")
    plt.plot(x, nn)


def cmatrix(labels, predict):
    cm = confusion_matrix(labels, predict)
    cm2 = cm / np.sum(cm, axis=1)
    cmp = ConfusionMatrixDisplay(
        confusion_matrix=cm2, display_labels=np.arange(constants.N_BACS))

    fig, ax = plt.subplots(figsize=(20, 20))
    cmp.plot(ax=ax, values_format='d')


def class_report(labels, predict):
    classfic = classification_report(
        labels, predict, np.arange(constants.N_BACS), constants.LABELS)
    print(classfic)


def plot_confusion_matrix(cm,
                          target_names,
                          customText=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          saveDir=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f} {}'.format(
        accuracy, misclass, customText))
    if saveDir is not None:
        plt.savefig(Join(saveDir, 'cm'))
    plt.show()


def plot_inference(img, predict, name, saveDir=None):
    I = np.dstack([img, img, img])
    rows, cols = img.shape
    pred_count = 0
    counter = 0

    for r in range(rows):
        for c in range(cols):
            if img[r, c] != 0:
                I[r, c, :] = constants.COLORS_RGB[constants.LABELS[predict[pred_count]]]
                pred_count += 1

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8)
    ax.set_title(name)
    for i in range(16):
        ax.plot(constants.COLORS_RGB[constants.LABELS[i]], label=constants.LABELS[i],
                linewidth=10, color=constants.COLORS_RGB[constants.LABELS[i]])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax.imshow(I, cmap=ListedColormap(COLORS_RGB.values()))
    if saveDir != None:
        plt.savefig(saveDir, bbox_inches=Bbox.from_extents(0, 0, 10, 8))


def make_inf(model, path):
    try:
        os.makedirs(path)
    except:
        print("Skipped - Directory already created!")

    metric_inference = Join(path, 'inference-{}')
    for i in range(constants.N_BACS):
        test_path = Join(constants.PREPROCESS_STORE, 'Test')
        bac_cube = helper.get_cube_by_index(test_path, i, 'masked.pickle')
        mat = helper.hsi2matrix(bac_cube)
        mat = helper.remove_blank_lines_for_one_bac(mat)
        mat = mat - np.mean(mat)

        classes = cnn.predict_on_bac(
            model, mat.reshape(mat.shape[0], mat.shape[1], 1))
        plot_inference(helper.get_layer(bac_cube, 1), classes,
                       constants.LABELS[i] + ' - {} espectros'.format(mat.shape[1]), metric_inference.format(constants.LABELS[i]))


def metrify(xtest, ytest, model, labels, saveDir=None):
    if saveDir is not None:
        try:
            os.makedirs(saveDir)
        except:
            print("Skipped - Directory already created!")

    a, b, c = model.evaluate(xtest, ytest, batch_size=1500)
    ypred = model.predict(xtest, batch_size=1800, verbose=1)
    classes = np.argmax(ypred, axis=1)
    lab = np.array(ytest, dtype=np.uint8)
    cm = confusion_matrix(lab, classes)
    plot_confusion_matrix(
        cm, labels, 'loss: {:0.4f} scc: {:0.4f}'.format(a, c), saveDir=saveDir)
    classfic = classification_report(
        lab, classes, labels=np.arange(16), target_names=labels, digits=5)
    print(classfic)
