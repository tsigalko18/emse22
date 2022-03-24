import csv
import os
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

import glob

import pandas as pd
from natsort import natsorted

from visual_matching.compute_similarities import get_match
from utils import load_training_set_from_csv_file, load_sdc_model, get_indexes_nominal_real, make_prediction

FROM_FILE = False


def main(model_filename, method, verbose):
    sdc_model = load_sdc_model(model_filename, is_sim=False)

    all_maes = []
    for i in range(1, 19):

        folder_to_match = "data\\rq1\\nominal-real-" + model_filename.lower() + "\\simulation" + str(i)

        print("running %s" % folder_to_match)

        image_files = natsorted(glob.glob(folder_to_match + "/*.jpg"))

        indexes = get_indexes_nominal_real(rq="rq1", model_name=model_filename, sim_id=i)

        df_train = load_training_set_from_csv_file(is_sim=False,
                                                   first=indexes[0],
                                                   last=indexes[1])

        if FROM_FILE:
            df_test = pd.read_csv(folder_to_match + ".csv")
        else:
            df_test = pd.DataFrame(image_files, columns=['image_path'])

        list_all_errors_in_subsequence = []
        list_all_predictions_in_subsequence = []
        list_all_ground_truths = []
        matches = 0

        for index_test, row_test in df_test.iterrows():
            if not FROM_FILE:
                idx, img, matched = get_match(df_train,
                                              row_test['image_path'],
                                              method=method,
                                              verbose=verbose,
                                              is_sim=False,
                                              model=model_filename)

                if matched:
                    matches = matches + 1
                else:
                    continue

            if FROM_FILE:
                ground_truth_steering_angle = row_test['steering']
            else:
                ground_truth_steering_angle = img[2]

            ground_truth_steering_angle = round(ground_truth_steering_angle, 3)
            list_all_ground_truths.append(ground_truth_steering_angle)

            if FROM_FILE:
                img = Image.open(os.path.join(folder_to_match, row_test.image_path))
            else:
                img = Image.open(os.path.join('data', 'train', 'tub320x240_train_real', img.image_path))

            if img is None:
                print("image not found")
                exit()

            sa = make_prediction(img, sdc_model)
            mae = abs(ground_truth_steering_angle - sa)

            list_all_predictions_in_subsequence.append(sa)

            if verbose:
                print("ground truth: %.3f" % ground_truth_steering_angle)
                print("prediction: %.3f" % sa)
                print("absolute error: %.3f\n" % mae)

            list_all_errors_in_subsequence.append(mae)

            # plots the images
            I1 = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", size=22)
            I1.text((0, 0), "label: " + str(round(ground_truth_steering_angle, 3)), font=font, fill=(0, 255, 0))
            I1.text((0, 20), "prediction: " + str(round(sa, 3)), font=font, fill=(255, 0, 0))

            image_file_id = row_test.image_path.replace(folder_to_match + "\\", "").split("_")[0]

            from pathlib import Path
            Path("temp2/" + str(i)).mkdir(parents=True, exist_ok=True)
            img.save("temp2/" + str(i) + "/" + str(image_file_id) + ".png")

        mae = round(np.mean(list_all_errors_in_subsequence), 3)
        t = 0.1
        print("MAE %s - above %s? %s" % (str(mae), str(t), mae > t))
        all_maes.append(mae)

    return all_maes


if __name__ == '__main__':
    STORE_CSV = True

    for model_name in ["AUTUMN", "CHAUFFEUR"]:
        print("using model %s" % model_name)
        for method in ["ssim"]:
            print("using method %s" % method)
            all_maes = main(model_name, method, verbose=False)
        print(all_maes)

        if STORE_CSV:
            with open('rq1-results-REAL-nominal' + '-' + model_name + '.csv', 'w',
                      newline='',
                      encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(["MAE"])
                writer.writerows(map(lambda x: [x], all_maes))

            f.close()
