import json

import cv2 as cv2
import numpy as np

from online_vs_offline_real import img_crop
from visual_matching.algorithms import mse, histogram_matching, psnr, ssim_skimage


def compute_similarity(im1, im2, method="mse"):
    if method is "mse":
        return mse(im1, im2)
    if method is "histogram":
        return histogram_matching(im1, im2)
    if method is "psnr":
        return psnr(im1, im2)
    if method is "ssim":
        return ssim_skimage(im1, im2)
    else:
        print("visual matching method unknown. Use one among [mse, histogram, psnr, ssim]")
        exit()


def get_match(df_train, img, method="ssim", verbose=False, is_sim=True, model='AUTUMN'):
    if verbose:
        print("matching image %s" % img)

    img_to_match = cv2.imread(img)

    if not is_sim:
        # crop only real world images
        img_to_match = img_crop(img_to_match, 100, 0)

    similarity_list = list()
    best_similarity = None

    for index_train, row_train in df_train.iterrows():
        if is_sim:
            training_img_to_be_matched = cv2.imread('data/train/tub320x240_train_sim/' + row_train.image_path)

            # if model == "AUTUMN":
            #     # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            #     training_img_to_be_matched = cv2.resize(training_img_to_be_matched, (200, 110))  # for DAVE-2

        else:
            training_img_to_be_matched = cv2.imread('data/train/tub320x240_train_real/' + row_train.image_path)
            training_img_to_be_matched = img_crop(training_img_to_be_matched, 100, 0)

        sim = compute_similarity(training_img_to_be_matched, img_to_match, method)
        sim = round(sim, 3)

        if method is "ssim":
            if sim >= 0.2:  # 0.30 for real; 0.50 for sim
                similarity_list.append(sim)
        elif method is "mse":
            if sim < 5000:
                similarity_list.append(sim)

    if len(similarity_list) == 0:
        print("No matches found with the given threshold")
        exit()

    if method is "mse":
        best_similarity = np.min(similarity_list)
    elif method in ["histogram", "psnr", "ssim"]:
        best_similarity = np.max(similarity_list)

    index_best = similarity_list.index(best_similarity)

    if verbose:
        if len(similarity_list) > 0:

            # load the best matched image
            if is_sim:
                print(
                    "\tmethod %s: best match is train image at index %d (%.2f): %s" % (
                        method, index_best, best_similarity,
                        'data/train/tub320x240_train_sim/' + str(df_train.iloc[index_best].image_path)))

                best_match_img = cv2.imread(
                    'data/train/tub320x240_train_sim/' + str(df_train.iloc[index_best].image_path))
                # if model == "AUTUMN":
                #     # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                #     best_match_img = cv2.resize(best_match_img, (200, 110))  # for DAVE-2
                # else:
                #     best_match_img = img_crop(best_match_img, 100, 0)
            else:
                print(
                    "\tmethod %s: best match is train image at index %d (%.2f): %s" % (
                        method, index_best, best_similarity,
                        'data/train/tub320x240_train_real/' + str(df_train.iloc[index_best].image_path)))

                best_match_img = cv2.imread(
                    'data/train/tub320x240_train_real/' + str(df_train.iloc[index_best].image_path))
                best_match_img = img_crop(best_match_img, 100, 0)

            # display side by side image to be matched with the best matched image
            numpy_horizontal_concat = np.concatenate((img_to_match, best_match_img), axis=1)
            cv2.imshow('image to be matched / best matched image / ' + str(best_similarity), numpy_horizontal_concat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        else:
            print("\timage non matched")

    return index_best, df_train.iloc[index_best], len(similarity_list) > 0


def get_steering_angles_from_prefailing_list(prefailing_list):
    steering_angles = []
    for j in prefailing_list:
        with open(j) as json_file:
            data = json.load(json_file)
            steering_angles.append(data["pilot/angle"])
            json_file.close()

    return steering_angles


def get_images_from_prefailing_list(prefailing_list):
    images = []
    for j in prefailing_list:
        with open(j) as json_file:
            data = json.load(json_file)
            images.append(data["cam/image_array"])
            json_file.close()

    return images


def get_match_by_steering_angle(df_train, sa):
    min = 100000
    min_image = ''
    gt = 100000

    for index_train, row_train in df_train.iterrows():
        sa_train = row_train.steering
        diff = abs(sa - sa_train)
        if diff < min:
            min = diff
            gt = sa_train
            min_image = row_train.image_path

    assert min_image != ''
    assert gt != 100000

    return min_image, gt


if __name__ == '__main__':
    print()
