import glob
import os

import numpy as np
import pandas as pd
from cv2 import cv2
from matplotlib import pyplot as plt
from natsort import natsorted
from tensorflow import keras

from online_vs_offline_real import normalize_and_crop
from visual_matching.compute_similarities import compute_similarity

TRAINING_SET_DIR = "data/train/"


def load_training_set_from_csv_file(is_sim=True, first=None, last=None):
    """
    loads in memory all images from the training set. If first & last parameters are not specified
    the whole training set is loaded, otherwise only the subset between first and last is retrieved
    :param is_sim: whether to load the file for simulated data or real-world otherwise
    :param first: id of the first frame for selection
    :param last: id of the last frame for selection
    :return:
    """

    # read all csv file and sort it by image_path using natural sort because they are strings
    if is_sim:
        df = pd.read_csv(TRAINING_SET_DIR + 'tub320x240_train_sim.csv')
        df = df.set_index('frameId')
        df.index = natsorted(df.index)
    else:
        df = pd.read_csv(TRAINING_SET_DIR + 'tub320x240_train_real.csv')
        df = df.set_index('frame_id')
        df.index = natsorted(df.index)

    if first is None and last is None:
        print("Loading the entire training set")
    elif (first is None and last is not None) or (first is not None and last is None):
        print("Incorrect subset values: Both first/last values should be specified")
        exit()
    elif first == last:
        print("Incorrect subset values: first and last indexes are the same")
        exit()
    else:
        print("Selecting subset between the indexes: [%d, %d]" % (first, last))
        if is_sim:
            df = df.iloc[first:last]
        else:
            if last - 690 < first - 690:
                print("Something wrong with the subsequence: last index is lower than first index")
                exit()
            else:
                # a = abs(first) + 690
                # b = 690 + abs(last)
                a = abs(first)
                b = abs(last)
                df = df.iloc[a:b]

    return df


def load_sdc_model(model_name, is_sim):
    filename = None

    if model_name == 'AUTUMN':
        if is_sim:
            filename = 'sdc-sim/autumn-sim.h5'
        else:
            filename = 'sdc-real/autumn-real.h5'
    elif model_name == 'CHAUFFEUR':
        if is_sim:
            filename = 'sdc-sim/chauffeur-sim.h5'
        else:
            filename = 'sdc-real/chauffeur-real.h5'
    else:
        print("model name unsupported. Possible choices are [AUTUMN|CHAUFFEUR]")

    # TODO: manage the case in which the filename/model does not exist
    sdc_model = keras.models.load_model(filename)
    print("loaded model %s" % filename)
    return sdc_model


def get_indexes_nominal_sim(rq, model_name, sim_id):
    print("using precomputed subsequences")
    if rq == "rq1":
        if model_name == "AUTUMN":
            indexes = [(0, 95), (0, 95), (0, 20), (0, 20), (0, 20), (260, 262), (260, 262), (260, 262), (260, 262),
                       (230, 240), (240, 250), (250, 260), (290, 300), (286, 300), (716, 725), (872, 881), (872, 890),
                       (750, 760)]
        else:
            indexes = [(1093, 1100), (1093, 1100), (1338, 1339), (1341, 1343), (1360, 1369), (1369, 1375), (1369, 1375),
                       (1369, 1380), (1369, 1390), (1369, 1400), (605, 606), (1369, 1370), (640, 641), (640, 641),
                       (640, 641), (1407, 1430), (748, 750), (748, 750)]

        return indexes[sim_id - 1]
    else:
        print("wrong rq id")
        exit()


def get_indexes_nominal_real(rq, model_name, sim_id):
    print("using precomputed subsequences")
    if rq == "rq1":
        if model_name == "AUTUMN":
            indexes = [(0, 95), (0, 95), (0, 20), (0, 20), (0, 20), (260, 265), (260, 265), (260, 262), (260, 262),
                       (288, 290), (288, 290), (286, 290), (286, 290), (286, 300), (872, 881), (872, 881), (872, 890),
                       (872, 881)]
        else:
            indexes = [(1093, 1100), (1093, 1100), (1338, 1339), (1341, 1343), (1360, 1369), (1369, 1375), (1369, 1375),
                       (1369, 1380), (1369, 1390), (1369, 1400), (1369, 1410), (1369, 1370), (1404, 1408), (1407, 1411),
                       (1407, 1420), (1407, 1430), (1407, 1411), (1407, 1411)]

        return indexes[sim_id - 1]
    else:
        print("wrong rq id")
        exit()


def get_indexes_prefailure_sim(rq, model_name, sim_id):
    print("using precomputed subsequences")
    if rq == "rq1":
        if model_name == "AUTUMN":
            indexes = [(250, 340), (290, 350), (292, 350), (280, 350), (280, 350), (292, 350), (292, 350), (198, 350),
                       (198, 350), (220, 250), (240, 300), (620, 690), (639, 690), (616, 664), (622, 667), (620, 667),
                       (650, 690), (650, 690)]
        else:
            indexes = [(220, 280), (234, 280), (225, 280), (275, 350), (280, 350), (1, 115), (625, 685), (538, 685),
                       (538, 685), (536, 570), (602, 646), (569, 622), (629, 654), (536, 570), (629, 690), (629, 645),
                       (629, 690), (629, 690)]

        return indexes[sim_id - 1]
    else:
        print("wrong rq id")
        exit()


def get_indexes_prefailure_real(rq, model_name, sim_id):
    print("using precomputed subsequences")
    if rq == "rq1":
        if model_name == "AUTUMN":
            indexes = [(40, 60), (60, 80), (800, 850), (850, 885), (850, 930), (935, 973), (968, 1014), (968, 1105),
                       (1100, 1140), (1130, 1210), (1170, 1383), (1200, 1317), (1317, 1471), (1487, 1557), (1550, 1600),
                       (1600, 1650), (10, 30), (1700, 1790)]

        else:
            indexes = [(719, 760), (750, 800), (800, 885), (870, 940), (935, 1005), (1015, 1095), (1095, 1250),
                       (322, 327), (1450, 1550), (1565, 1655), (1787, 1838), (1830, 1880), (1927, 1990), (1928, 1990),
                       (2008, 2074), (2137, 2225), (2243, 2372), (2395, 2483)]

        return indexes[sim_id - 1]
    else:
        print("wrong rq id")
        exit()


def make_prediction(img, sdc_model):
    img_arr = np.array(img)
    img_arr = normalize_and_crop(img_arr, crop=100)
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    prediction = sdc_model.predict(img_arr)
    sa = prediction[0]
    sa = sa[0][0]
    temp = 1.0 if sa > 1.0 else sa
    sa = temp
    temp = -1.0 if sa < -1.0 else sa
    sa = temp
    sa = round(sa, 3)
    return sa


def sort_donkey_csv_filenames():
    for file in ["DAVE2-MC-REAL-Run1.csv",
                 "DAVE2-MC-REAL-Run2.csv",
                 "DAVE2-MC-REAL-Run3.csv",
                 "DAVE2-MC-REAL-Run4.csv",
                 "DAVE2-MC-REAL-Run5.csv",
                 "DAVE2-MC-REAL-Run6.csv"]:
        df = pd.read_csv(file)

        df.index = df.image_path

        df = df.reindex(index=natsorted(df.index))

        df.to_csv(file + '-sorted.csv', index=False)

    # extension = 'csv'
    # all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #
    # # for files in ['DAVE2-MC-REAL-Run1.csv',
    # #               "DAVE2-MC-REAL-Run2.csv",
    # #               "DAVE2-MC-REAL-Run3.csv",
    # #               "DAVE2-MC-REAL-Run4.csv",
    # #               "DAVE2-MC-REAL-Run5.csv",
    # #               "DAVE2-MC-REAL-Run6.csv"]:
    #
    # combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # # export to csv
    # combined_csv.to_csv("DAVE2-REAL-50Laps.csv", index=False, encoding='utf-8-sig')


def rq1_threshold_sim_pseudosim():
    is_sim = True

    if is_sim:
        sim_folder = "data/test/sim-test1"  # sim
        pseudosim_folder = "data/test/sim-test2"  # sim -> real -> sim MATCHING
        non_matching_folder = "data/test/sim-test3"  # sim -> real -> sim NON MATCHING
    else:
        sim_folder = "data/test/real-test1"  # sim
        pseudosim_folder = "data/test/real-test2"  # sim -> real -> sim MATCHING
        non_matching_folder = "data/test/real-test3"  # sim -> real -> sim NON MATCHING

    sim_image_files = natsorted(glob.glob(sim_folder + "/*.jpg"))
    pseudosim_image_files = natsorted(glob.glob(pseudosim_folder + "/*.jpg"))
    non_matching_image_files = natsorted(glob.glob(non_matching_folder + "/*.jpg"))

    assert len(sim_image_files) == len(pseudosim_image_files) == len(non_matching_image_files)

    ssim_distances = []
    for idx in range(len(sim_image_files)):
        # print(idx)
        sim_img = cv2.imread(os.path.join(sim_image_files[idx]))
        sim_img = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
        sim_img = sim_img[100:, :]
        sim_img = cv2.resize(sim_img, (120, 160))

        pseudosim_img = cv2.imread(os.path.join(pseudosim_image_files[idx]))

        ssim_distances.append(compute_similarity(sim_img, pseudosim_img, method='ssim'))

    ssim_distances_non_matching = []
    for idx in range(len(sim_image_files)):
        # print(idx)
        sim_img = cv2.imread(os.path.join(sim_image_files[idx]))
        sim_img = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
        sim_img = sim_img[100:, :]
        sim_img = cv2.resize(sim_img, (120, 160))

        non_matching_img = cv2.imread(os.path.join(non_matching_image_files[idx]))
        non_matching_img = cv2.cvtColor(non_matching_img, cv2.COLOR_BGR2RGB)
        non_matching_img = non_matching_img[100:, :]
        non_matching_img = cv2.resize(non_matching_img, (120, 160))

        ssim_distances_non_matching.append(compute_similarity(sim_img, non_matching_img, method='ssim'))

    plt.hist(ssim_distances, label='matching')
    plt.hist(ssim_distances_non_matching, label='non matching')
    plt.title("ssim sim" if is_sim else "ssim real")
    plt.legend(loc='upper right')
    plt.show()
    plt.close()

    print("min ssim matching %.2f" % np.min(ssim_distances))
    print("max ssim non matching %.2f" % np.max(ssim_distances_non_matching))

    # threshold visual similarity real SSIM > 0.50
    # threshold visual similarity real MSE < 2000

    # threshold visual similarity sim SSIM > 0.68
    # threshold visual similarity sim MSE < 1500
