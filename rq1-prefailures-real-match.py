import csv
import os
import warnings

from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import glob

import numpy as np
import pandas as pd
from natsort import natsorted

from visual_matching.compute_similarities import get_match
from utils import load_training_set_from_csv_file, get_indexes_prefailure_real, make_prediction, load_sdc_model


def main(model_filename, method, verbose):
    sdc_model = load_sdc_model(model_filename, is_sim=False)

    all_maes = []
    for i in range(1, 19):

        folder_to_match = "data/rq1/pre-fail-real-" + model_filename.lower() + "/simulation" + str(i)

        print("running %s" % folder_to_match)

        image_files = natsorted(glob.glob(folder_to_match + "/*.jpg"))

        indexes = get_indexes_prefailure_real(rq="rq1", model_name=model_filename, sim_id=i)

        df_train = load_training_set_from_csv_file(is_sim=False,
                                                   first=indexes[0],
                                                   last=indexes[1])

        df_test = pd.DataFrame(image_files, columns=['image_path'])

        list_all_errors_in_subsequence = []
        list_all_ground_truth_in_subsequence = []
        matches = 0

        for index_test, row_test in df_test.iterrows():
            i, img, matched = get_match(df_train,
                                        row_test['image_path'],
                                        method=method,
                                        verbose=verbose,
                                        is_sim=False,
                                        model=model_filename)

            if matched:
                matches = matches + 1
            else:
                continue

            ground_truth_steering_angle = img[2]

            img = Image.open(os.path.join('data/train/tub320x240_train_real/', img.image_path))

            sa = make_prediction(img, sdc_model)
            mae = abs(ground_truth_steering_angle - sa)

            if verbose:
                print("ground thruth: %.2f" % ground_truth_steering_angle)
                print("prediction: %.2f" % sa)
                print("absolute error: %.2f" % mae)

            list_all_errors_in_subsequence.append(mae)
            list_all_ground_truth_in_subsequence.append(ground_truth_steering_angle)

        if verbose:
            print("matches %d/%d (%d %%)" % (matches, len(df_test), matches / len(df_test) * 100))

        mae = round(np.mean(list_all_errors_in_subsequence), 3)
        t = 0.1
        print("MAE %s - above %s? %s" % (str(mae), str(t), mae > t))
        all_maes.append(mae)

    return all_maes


if __name__ == '__main__':
    STORE_CSV = True
    # for model_name in ["AUTUMN", "CHAUFFEUR"]:
    for model_name in ["CHAUFFEUR"]:
        print("using model %s" % model_name)

        for method in ["ssim"]:
            print("using method %s" % method)
            all_maes = main(model_name, method, verbose=False)

        print(all_maes)

        if STORE_CSV:
            with open('rq1-results-REAL-pre-fail' + '-' + model_name + '.csv', 'a',
                      newline='',
                      encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(["MAE"])
                writer.writerows(map(lambda x: [x], all_maes))

            f.close()
