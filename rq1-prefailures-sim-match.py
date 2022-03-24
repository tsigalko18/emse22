import csv
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import glob
import json

import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
from visual_matching.compute_similarities import get_match
from utils import load_training_set_from_csv_file, load_sdc_model, get_indexes_prefailure_sim, make_prediction


def main(model_filename, method, seq, verbose):
    sdc_model = load_sdc_model(model_filename, is_sim=True)

    all_maes = []
    for i in range(1, 19):

        folder_to_match = ""

        if model_name == 'AUTUMN':
            folder_to_match = "data/rq1/pre-fail-sim-autumn/simulation" + str(i) + "/tub_1_22-01-18"
        elif model_name == 'CHAUFFEUR':
            folder_to_match = "data/rq1/pre-fail-sim-chauffeur/simulation" + str(i) + "/tub_1_22-01-24"

        print("running %s" % folder_to_match)
        FPS = 20
        SECONDS = 3
        L = FPS * SECONDS  # sequence length (20 = 1 sec of simulation)

        image_files = natsorted(glob.glob(folder_to_match + "/*.jpg"))
        json_files = natsorted(glob.glob(folder_to_match + "/record_*.json"))

        first_failing_frame = -1
        for idx, json_file in enumerate(json_files):
            with open(json_file) as json_file_dump:
                data = json.load(json_file_dump)
                current_cte = data['cte']
                if current_cte is not None and abs(current_cte) > 2.2:
                    first_failing_frame = idx
                    print("first out of track frame is number %d: %s" % (first_failing_frame, data['cam/image_array']))
                    break

        if seq == "ENTIRE":
            first_prefailing_frame = 0
            prefailing_list_of_images = image_files[first_prefailing_frame:first_failing_frame]
        else:
            if L > first_failing_frame:  # sequence is too short, we take it all
                first_prefailing_frame = 0
            else:
                first_prefailing_frame = first_failing_frame - L
                assert (first_prefailing_frame > 1)

            prefailing_list_of_images = image_files[first_prefailing_frame:first_failing_frame]

            if L <= first_failing_frame:
                assert len(prefailing_list_of_images) == L

        print("simulation {}: sequence {}-{}".format(i, first_prefailing_frame, first_failing_frame))
        # continue

        steering_angles = []
        for idx, json_file in enumerate(json_files):
            with open(json_file) as json_file_dump:
                data = json.load(json_file_dump)
                steering_angle = data['pilot/angle']
                steering_angles.append(steering_angle)

        prefailing_list_of_steering_angles = steering_angles[first_prefailing_frame:first_failing_frame]

        indexes = get_indexes_prefailure_sim(rq="rq1", model_name=model_filename, sim_id=i)

        df_train = load_training_set_from_csv_file(is_sim=True,
                                                   first=indexes[0],
                                                   last=indexes[1])

        df_test = pd.DataFrame(prefailing_list_of_images, columns=['image_path'])

        list_all_errors_in_subsequence = []

        for index_test, row_test in df_test.iterrows():
            i, img, matched = get_match(df_train,
                                        row_test['image_path'],
                                        method=method,
                                        verbose=verbose,
                                        is_sim=True,
                                        model=model_name)

            ground_truth_steering_angle = img[2]

            # online vs online matching
            # sa = prefailing_list_of_steering_angles[index_test]

            # online vs offline matching
            img = Image.open(os.path.join('data/train/tub320x240_train_sim/', img.image_path))
            sa = make_prediction(img, sdc_model)

            sa = round(sa, 3)
            mae = abs(ground_truth_steering_angle - sa)

            if verbose:
                print("ground truth: %.2f" % ground_truth_steering_angle)
                print("prediction: %.2f" % sa)
                print("absolute error: %.2f" % mae)

            list_all_errors_in_subsequence.append(mae)

        mae = round(np.mean(list_all_errors_in_subsequence), 3)
        t = 0.1
        print("MAE %s - above %s? %s" % (str(mae), str(t), mae > t))
        all_maes.append(mae)

    return all_maes


if __name__ == '__main__':

    STORE_CSV = False

    for model_name in ['CHAUFFEUR']:
        print("using model %s" % model_name)
        for method in ["ssim"]:
            print("using method %s" % method)
            all_maes = main(model_name, method, "", verbose=False)
        print(all_maes)


        if STORE_CSV:
            with open('rq1-results-SIMULATED-prefail' + '-' + model_name + '.csv', 'w',
                      newline='',
                      encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(["MAE"])
                writer.writerows(map(lambda x: [x], all_maes))

            f.close()
