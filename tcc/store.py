import pickle
import os


def save_pickle(path, filename, p):
    pickle_out = open(os.path.join(path, filename), "wb")
    pickle.dump(p, pickle_out)
    pickle_out.close()


def save_all(path, calib, masked):
    try:
        os.makedirs(path)
    except:
        print("Skipped - Directory already created!")

    save_pickle(path, 'calib.pickle', calib)
    save_pickle(path, 'masked.pickle', masked)


def load_pickle(filename, dirpath):
    path = os.path.join(dirpath, filename)
    pickle_in = open(path, "rb")
    return pickle.load(pickle_in)
