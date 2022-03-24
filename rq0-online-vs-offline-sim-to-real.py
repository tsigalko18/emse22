import warnings

warnings.filterwarnings("ignore")

import glob
from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted
import json
import tensorflow as tf
from tensorflow.python import keras
import csv
from statistical_tests.wilcoxon_test import *
import numpy as np


def compute_offline_steering_angles(model, start, end, type):
    sdc_model = None

    if model == "DAVE-2-real":
        sdc_model = keras.models.load_model('sdc-real/linear_andrea_tub_2.h5')
        # sdc_model = keras.models.load_model('sdc-real/dave2-real.h5')
    elif model == "CHAUFFEUR-real":
        sdc_model = keras.models.load_model('sdc-real/chauffeur-real.h5')
    else:
        print("Unknown SDC model. Exiting.")
        exit()

    offline_steering_angles = []

    if os.path.exists('predictions_' + model + '.npy'):
        offline_steering_angles = np.load('predictions_' + model + '.npy')
        print("loaded predictions_" + model + ".npy")
    else:
        retained = []
        for img_name in image_files:
            image_file_id = img_name.replace(test_folder, "").split("_")[0]
            if start <= int(image_file_id) <= end:
                retained.append(img_name)

        inputs = load_and_preprocess_images(retained, model, type)

        for img_arr in inputs:
            img_arr = img_arr.reshape((1,) + img_arr.shape)

            with tf.device('/CPU:0'):
                prediction = sdc_model.predict(img_arr)

            sa = prediction[0]
            sa = sa[0][0]

            temp = 1.0 if sa > 1.0 else sa
            sa = temp
            temp = -1.0 if sa < -1.0 else sa
            sa = temp

            sa = round(sa, 3)
            offline_steering_angles.append(sa)

    return offline_steering_angles


def compute_statistics(online_steering_angles, offline_steering_angles):
    simulation1 = np.asarray(online_steering_angles, dtype=float)
    simulation2 = np.asarray(offline_steering_angles, dtype=float)

    w_statistic, pvalue = wilcoxon(simulation1, simulation2)
    cohensd = cohend(simulation1, simulation2)
    mae = abs((simulation1 - simulation2).mean(axis=None))
    mae_in_deg = mae * 16
    print("mae offline vs online: %.2f\t%.2f deg" % (mae, mae_in_deg))
    print(f"P-Value is: {pvalue}")
    print(f"Cohen's D is: {cohensd}")

    diff_stat_sign = "Distributions are statistically different\n" if pvalue <= 0.05 else f"Distributions are statistically the same\n"
    print(diff_stat_sign)

    pow = run_power_analysis_two_sets(simulation1, simulation2)

    return mae, mae_in_deg, pvalue, cohensd[0], pow


def retrieve_online_steering_angles(image_files, test_folder, start, end):
    online_steering_angles = []
    for i in image_files:
        image_file_id = i.replace(test_folder, "").split("_")[0]
        if start <= int(image_file_id) <= end:
            with open(test_folder + "/record_" + image_file_id + ".json") as json_file:
                online_steering_angle = json.load(json_file)["user/angle"]
                # print(online_steering_angle)
                online_steering_angles.append(online_steering_angle)
    return online_steering_angles


def img_crop(img_arr, top, bottom):
    if bottom is 0:
        end = img_arr.shape[0]
    else:
        end = -bottom
    return img_arr[top:end, ...]


def normalize_and_crop(img_arr, ROI_CROP_TOP):
    img_arr = img_arr.astype(np.uint8) * 1.0 / 255.0
    img_arr = img_crop(img_arr, ROI_CROP_TOP, 0)
    if len(img_arr.shape) == 2:
        img_arrH = img_arr.shape[0]
        img_arrW = img_arr.shape[1]
        img_arr = img_arr.reshape(img_arrH, img_arrW, 1)
    return img_arr


def rgb2gray(rgb):
    '''
    take a numpy rgb image return a new single channel image converted to greyscale
    '''
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def load_scaled_image_arr(filename, height, width, crop_top):
    '''
    load an image from the filename, and use the cfg to resize if needed
    also apply cropping and normalize
    '''
    try:
        img = Image.open(filename)

        if img.height != height or img.width != width:
            img = img.resize((width, height))
        img_arr = np.array(img)
        img_arr = normalize_and_crop(img_arr, crop_top)
    except Exception as e:
        print(e)
        print('failed to load image:', filename)
        img_arr = None
    return img_arr


def load_and_preprocess_images(image_files, model, type):
    inputs = []
    print("\nCollecting {} test image paths ...".format(len(image_files)))
    for img_name in image_files:
        img = Image.open(os.path.join(img_name))
        img = img.convert('RGB')

        if model == 'DAVE-2-real':
            # settings for linear_andrea_tub_2.h5
            crop = 100
            img = img.resize((320, 240))
        elif model == 'CHAUFFEUR-real':
            crop = 120
            img = img.resize((320, 240))

        img = np.array(img)
        img = normalize_and_crop(img, crop)

        # Appending
        inputs.append(img)
    print("{} images paths collected.".format(len(inputs)))
    inputs = np.array(inputs)
    return inputs


def compare_online_vs_offline_steering_angles(image_files, test_folder, start, end, online_steering_angles,
                                              offline_steering_angles, name, type):
    retained = []
    for img_name in image_files:
        image_file_id = img_name.replace(test_folder, "").split("_")[0]
        if start <= int(image_file_id) <= end:
            retained.append(img_name)

    inputs = load_and_preprocess_images(retained, model, type)

    i = 0
    for img_arr in inputs:
        img_arr = img_arr.reshape((1,) + img_arr.shape)

        img_arr = img_arr[0, :, :, :]

        img = Image.fromarray((img_arr * 255).astype(np.uint8), 'RGB')

        I1 = ImageDraw.Draw(img)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=22)
        I1.text((0, 0), "label: " + str(round(online_steering_angles[i], 3)), font=font, fill=(0, 255, 0))
        I1.text((0, 20), "prediction: " + str(round(offline_steering_angles[i], 3)), font=font, fill=(255, 0, 0))

        # img.show()
        from pathlib import Path
        Path("temp/" + name).mkdir(parents=True, exist_ok=True)
        img.save("temp/" + name + "/" + str(i) + ".png")

        i = i + 1


def plot_online_vs_offline_steering_angles(online_steering_angles, offline_steering_angles, name):
    plt.plot(range(len(online_steering_angles)), online_steering_angles, '-k', color='green')
    plt.plot(range(len(online_steering_angles)), offline_steering_angles, '-k', color='red', alpha=0.3)
    plt.title(name)
    plt.savefig("temp/" + name.replace(" ", "-") + '.png')
    plt.clf()
    plt.show()


if __name__ == '__main__':

    INPUT_IMAGE_TYPE = "PSEUDOREAL"
    STORE_CSV = True

    if INPUT_IMAGE_TYPE == "PSEUDOREAL":
        test_folder = os.path.join("data", "rq0", "scenarios-pseudoreal/")
        print("Using input image type REAL")

    start_indexes = [1, 117, 235, 353, 471, 589, 707, 825, 943, 1061, 1179, 1297, 1415, 1533, 1651, 1769, 1887, 2005,
                     2123, 2241, 2359, 2477, 2595, 2713, 2831, 2949, 3067, 3185, 3303, 3421, 3539]

    end_indexes = [116, 234, 352, 470, 588, 706, 824, 942, 1060, 1178, 1296, 1414, 1532, 1650, 1768, 1886, 2004, 2122,
                   2240, 2358, 2476, 2594, 2712, 2830, 2948, 3066, 3184, 3302, 3420, 3538, 3656]

    image_files = natsorted(glob.glob(test_folder + "/*.jpg"))

    if STORE_CSV:
        header = ["MAE", "p-value", "effsize", "pow"]

    for model in ["DAVE-2-real", "CHAUFFEUR-real"]:

        if STORE_CSV:
            with open('rq0-results-' + INPUT_IMAGE_TYPE + '-' + model + '.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            f.close()

        for scenario in range(1, 2):
            print("Evaluating scenario %d" % scenario)

            # retrieves the online steering angles made by humans
            online_steering_angles = retrieve_online_steering_angles(image_files,
                                                                     test_folder,
                                                                     start_indexes[scenario - 1],
                                                                     end_indexes[scenario - 1])

            row = []

            print("Evaluating model %s" % model)

            # computes offline steering angles (DNN predictions) on the images retrieved online
            offline_steering_angles = compute_offline_steering_angles(model,
                                                                      start_indexes[scenario - 1],
                                                                      end_indexes[scenario - 1],
                                                                      INPUT_IMAGE_TYPE)

            assert len(online_steering_angles) == len(offline_steering_angles)

            name = model + '-scenario-' + str(scenario) + '-' + INPUT_IMAGE_TYPE
            # compares the online/offline steering angles for each image and stores them as annotated files
            compare_online_vs_offline_steering_angles(image_files,
                                                      test_folder,
                                                      start_indexes[scenario - 1],
                                                      end_indexes[scenario - 1],
                                                      online_steering_angles,
                                                      offline_steering_angles,
                                                      name, INPUT_IMAGE_TYPE)

            # statistical tests
            mae, mae_in_deg, pvalue, effsize, pow = compute_statistics(online_steering_angles, offline_steering_angles)

            row.append(round(mae, 2))
            row.append(pvalue)
            row.append(effsize)
            row.append(round(pow, 2))

            name = model + ' scenario ' + str(scenario) + ' ' + INPUT_IMAGE_TYPE + ' (MAE = ' + str(round(mae, 2)) + ')'

            # plots the distributions
            plot_online_vs_offline_steering_angles(online_steering_angles,
                                                   offline_steering_angles,
                                                   name)

            if STORE_CSV:
                with open('rq0-results-' + INPUT_IMAGE_TYPE + '-' + model + '.csv', 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                f.close()
