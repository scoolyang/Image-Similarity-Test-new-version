"""
Created on Mon Aug 12 15:37:14 2019

@author: cheng
"""
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

# This function is used to find out corresbounding imgae data given camera index and image number from all data txt file
# Three Inputs: 1. input YOLO output txtfile
#               2. image from which camera holder (ex: infinite, nissan)
#               3. input image number (the six digits of each image)
#               4. number of features need to be save in the matrix (default is 8)

# Two Outputs: 1. input matrix coming from YOLO output file
#               2. object to idx dictionary

def find_sample_data(open_file_name, camera_holder, img_num, features):
    save_object_to_idx_path = 'image_similarity_object_to_idx.npy'
    if not os.path.exists(save_object_to_idx_path):
        object_to_idx = {}
        np.save(save_object_to_idx_path, object_to_idx)
    input_x_matrix = np.zeros(features).reshape(1, features)
    if camera_holder == Argument['txt_file_camera_holder1']:
        # print(1)
        img_num_range_lb = 25
        img_num_range_ub = 31
        # Sample Test
        # img_num_range_lb = 21
        # img_num_range_ub = 26
    elif camera_holder == Argument['txt_file_camera_holder2']:
        img_num_range_lb = 23
        img_num_range_ub = 29
        # Sample Test
        # img_num_range_lb = 21
        # img_num_range_ub = 26
    with open(open_file_name) as file:
        data = file.readlines()
        new_image_flag = 0
        object_to_idx = np.load(save_object_to_idx_path)
        new_class_index = len(object_to_idx.item())
        for line in data:
            numbers = line.split()
            if numbers != []:
                if numbers[0][0:3] == 'exp' and numbers[0][img_num_range_lb:img_num_range_ub] == str(img_num):
                    # image_name = numbers[0][14:len(numbers[0])]
                    new_image_flag = 1
                    new_image_array = np.zeros([features]).reshape(1, features)
                elif numbers[0][0:3] == 'exp' and numbers[0][img_num_range_lb:img_num_range_ub] != str(img_num):
                    new_image_flag = 0
                if new_image_flag == 1 and numbers[0][0] != 'e':
                    test_digit = numbers[0]

                    if test_digit.isdigit():
                        detect_num = float(numbers[0])
                        x_min = float(numbers[1])
                        y_min = float(numbers[2])
                        width = float(numbers[3])
                        length = float(numbers[4])
                        class_label_string = numbers[5]
                        if class_label_string[-1] == ':':
                            class_label_string = class_label_string[0:(len(class_label_string) - 1)]
                        if class_label_string not in object_to_idx.item():
                            object_to_idx.item()[class_label_string] = new_class_index
                            np.save(save_object_to_idx_path, object_to_idx.item())
                            new_class_index += 1
                        cfs = numbers[6]
                        if cfs[-1] != '%':
                            cfs = numbers[7]
                    else:
                        detect_num += 1
                        class_label_string = numbers[0]
                        if class_label_string[-1] == ':':
                            class_label_string = class_label_string[0:(len(class_label_string) - 1)]
                        if class_label_string not in object_to_idx.item():
                            object_to_idx.item()[class_label_string] = new_class_index
                            np.save(save_object_to_idx_path, object_to_idx.item())
                            new_class_index += 1
                        cfs = numbers[1]
                        if cfs[-1] != '%':
                            cfs = numbers[2]

                    new_image_array[0, 0] = float(img_num)
                    new_image_array[0, 1] = detect_num
                    new_image_array[0, 2] = x_min
                    new_image_array[0, 3] = y_min
                    new_image_array[0, 4] = width
                    new_image_array[0, 5] = length
                    new_image_array[0, 6] = object_to_idx.item()[class_label_string]
                    new_image_array[0, 7] = float(cfs[0:2])
                    if cfs[0:3] == '100':
                        new_image_array[0, 7] = float(cfs[0:3])
                    input_x_matrix = np.concatenate((input_x_matrix, new_image_array), axis=0)

        input_x_matrix = np.delete(input_x_matrix, (0), axis=0)
    return input_x_matrix, object_to_idx

# This function used to find out desired object bouning box coordinates in an image
# Three Inputs: 1. input matrix generated from YOLO output txtfile
#               2. corresponding input image in rgb format
#               3. the desired number of object in the input image

# Four Outputs: 1. bounding box x axis min coordinate
#               2. bounding box x axis max coordinate
#               3. bounding box y axis min coordinate
#               4. bounding box y axis max coordinate

def find_coor(input_x_matrix, input_img, object_num):
    xmin = int(input_x_matrix[object_num][2])
    if xmin <= 0:
        xmin = 0
    xmax = xmin + int(input_x_matrix[object_num][4])
    ymin = int(input_x_matrix[object_num][3])
    ymax = ymin + int(input_x_matrix[object_num][5])
    if ymax >= input_img.shape[0]:
        ymax = input_img.shape[0]   #480
        # img_test.shape[1]  # 640
    return xmin, xmax, ymin, ymax

# This function used to find out all the HSV histegram of objects in the input image
# Three Inputs: 1. input matrix generated from YOLO output txtfile
#               2. corresponding input image in rgb format

# Three Outputs: 1. Hue histogram of each object as a list
#               2. Saturation  histogram of each obejct as a list
#               3. Value histogram of each object as a list

def find_hsv_hist(input_x_matrix, img_test):
    object_num_in_img = input_x_matrix.shape[0]
    hsv = cv.cvtColor(img_test, cv.COLOR_BGR2HSV)
    histSize = 256

    h_hist_array = np.zeros(180).reshape(180, 1)
    s_hist_array = np.zeros(histSize).reshape(histSize, 1)
    v_hist_array = np.zeros(histSize).reshape(histSize, 1)

    for k in range(object_num_in_img):
        xmin, xmax, ymin, ymax = find_coor(input_x_matrix, img_test, k)
        roi_img = hsv[ymin: ymax, xmin: xmax]
        hsv_planes = cv.split(roi_img)

        h_hist = cv.calcHist(hsv_planes, [0], None, [180], [0, 180])
        s_hist = cv.calcHist(hsv_planes, [1], None, [256], [0, 256])
        v_hist = cv.calcHist(hsv_planes, [2], None, [256], [0, 256])
        #
        new_h_hist = h_hist / np.amax(h_hist)
        new_s_hist = s_hist / np.amax(s_hist)
        new_v_hist = v_hist / np.amax(v_hist)

        h_hist_array = np.concatenate((h_hist_array, new_h_hist), axis=1)
        s_hist_array = np.concatenate((s_hist_array, new_s_hist), axis=1)
        v_hist_array = np.concatenate((v_hist_array, new_v_hist), axis=1)
        # hist = cv.calcHist([roi_img], [0, 1], None, [180, 256], [0, 180, 0, 256])
        # plt.imshow(hist, interpolation='nearest')
        # plt.show()
    h_hist_array = np.delete(h_hist_array, (0), axis=1).reshape(180, object_num_in_img)
    s_hist_array = np.delete(s_hist_array, (0), axis=1).reshape(256, object_num_in_img)
    v_hist_array = np.delete(v_hist_array, (0), axis=1).reshape(256, object_num_in_img)

    return h_hist_array, s_hist_array, v_hist_array

# This function used to find out two similarity obejcts in two different images based on HSV image
# Four Inputs: 1. input matrix generated from YOLO output txtfile of image 1
#              2. corresponding input image in rgb format of image 1
#              3. input matrix generated from YOLO output txtfile of image 2
#              4. corresponding input image in rgb format of image 2

# Three Outputs: 1. Pair list of each object in image 1 and the most similarity object in image 2
#                2. Dictionary contains the score of each pair
#                3. List contains the difference value between two lowest score

def find_hsv_similarity(input_x_matrix, input_x_img, input_y_matrix, input_y_img):
    object_num_in_x_img = input_x_matrix.shape[0]
    object_num_in_y_img = input_y_matrix.shape[0]

# Only Value is considered
#     x_v = find_hsv_hist(input_x_matrix, input_x_img)
#     y_v = find_hsv_hist(input_y_matrix, input_y_img)

# HSV are all considered
    x_h, x_s, x_v = find_hsv_hist(input_x_matrix, input_x_img)
    y_h, y_s, y_v = find_hsv_hist(input_y_matrix, input_y_img)

    hist_match_list = []
    hist_pair_score_dict = {}
    least_two_diff_list = []

    for i in range(object_num_in_x_img):
        sum_diff_list = []
        for j in range(object_num_in_y_img):
            cur_v_compareHist = chi_square(x_v[:, i].reshape(256, 1),
                                           y_v[:, j].reshape(256, 1))
            cur_h_compareHist = chi_square(x_h[:, i].reshape(180, 1),
                                           y_h[:, j].reshape(180, 1))
            cur_s_compareHist = chi_square(x_s[:, i].reshape(256, 1),
                                           y_s[:, j].reshape(256, 1))
            cur_total_Hist = cur_h_compareHist + cur_v_compareHist + cur_s_compareHist
            sum_diff_list.append(cur_total_Hist)
        min_sum_object = sum_diff_list.index(min(sum_diff_list))
        diff_x = find_diff_list_between_two_least_num(sum_diff_list)
        match_pair = [i, min_sum_object]
        hist_match_list.append(match_pair)
        least_two_diff_list.append(diff_x)
        if (i, min_sum_object) not in hist_pair_score_dict:
            hist_pair_score_dict[(i, min_sum_object)] = np.min(sum_diff_list)

        x1min, x1max, y1min, y1max = find_coor(input_x_matrix, input_x_img, i)
        roi_img1 = input_x_img[y1min: y1max, x1min: x1max]

        x2min, x2max, y2min, y2max = find_coor(input_y_matrix, input_y_img, min_sum_object)
        roi_img2 = input_y_img[y2min: y2max, x2min: x2max]

        bgr_image_x1 = roi_img1[..., ::-1]
        bgr_image_x2 = roi_img2[..., ::-1]

        # f = plt.figure()
        # f.add_subplot(1, 2, 1)
        # plt.imshow(bgr_image_x1)
        # f.add_subplot(1, 2, 2)
        # plt.imshow(bgr_image_x2)
        # plt.show(block=True)
    return hist_match_list, hist_pair_score_dict, least_two_diff_list

# This function used to find out two similarity obejcts in two different images based on H value only
# Four Inputs: 1. input matrix generated from YOLO output txtfile of image 1
#              2. corresponding input image in rgb format of image 1
#              3. input matrix generated from YOLO output txtfile of image 2
#              4. corresponding input image in rgb format of image 2

# Three Outputs: 1. Pair list of each object in image 1 and the most similarity object in image 2
#                2. Dictionary contains the score of each pair
#                3. List contains the difference value between two lowest score

def find_h_similarity(input_x_matrix, input_x_img, input_y_matrix, input_y_img):
    object_num_in_x_img = input_x_matrix.shape[0]
    object_num_in_y_img = input_y_matrix.shape[0]

# Only Value is considered
#     x_v = find_hsv_hist(input_x_matrix, input_x_img)
#     y_v = find_hsv_hist(input_y_matrix, input_y_img)

# HSV are all considered
    x_h, x_s, x_v = find_hsv_hist(input_x_matrix, input_x_img)
    y_h, y_s, y_v = find_hsv_hist(input_y_matrix, input_y_img)

    hist_match_list = []
    hist_pair_score_dict = {}
    least_two_diff_list = []

    for i in range(object_num_in_x_img):
        sum_diff_list = []
        for j in range(object_num_in_y_img):
            # cur_v_compareHist = chi_square(x_v[:, i].reshape(256, 1),
            #                                y_v[:, j].reshape(256, 1))
            cur_h_compareHist = chi_square(x_h[:, i].reshape(180, 1),
                                           y_h[:, j].reshape(180, 1))
            # cur_s_compareHist = chi_square(x_s[:, i].reshape(256, 1),
            #                                y_s[:, j].reshape(256, 1))
            cur_total_Hist = cur_h_compareHist# + cur_h_compareHist + cur_s_compareHist
            sum_diff_list.append(cur_total_Hist)
        min_sum_object = sum_diff_list.index(min(sum_diff_list))
        diff_x = find_diff_list_between_two_least_num(sum_diff_list)
        match_pair = [i, min_sum_object]
        hist_match_list.append(match_pair)
        least_two_diff_list.append(diff_x)
        if (i, min_sum_object) not in hist_pair_score_dict:
            hist_pair_score_dict[(i, min_sum_object)] = np.min(sum_diff_list)

        x1min, x1max, y1min, y1max = find_coor(input_x_matrix, input_x_img, i)
        roi_img1 = input_x_img[y1min: y1max, x1min: x1max]

        x2min, x2max, y2min, y2max = find_coor(input_y_matrix, input_y_img, min_sum_object)
        roi_img2 = input_y_img[y2min: y2max, x2min: x2max]

        bgr_image_x1 = roi_img1[..., ::-1]
        bgr_image_x2 = roi_img2[..., ::-1]

        # f = plt.figure()
        # f.add_subplot(1, 2, 1)
        # plt.imshow(bgr_image_x1)
        # f.add_subplot(1, 2, 2)
        # plt.imshow(bgr_image_x2)
        # plt.show(block=True)
    return hist_match_list, hist_pair_score_dict, least_two_diff_list

# This function used to find out all the BGR histegram of objects in the input image
# Three Inputs: 1. input matrix generated from YOLO output txtfile
#               2. corresponding input image in rgb format

# Three Outputs: 1. Blue histogram of each object as a list
#               2. Green histogram of each obejct as a list
#               3. Red histogram of each object as a list

def find_bgr_hist(input_x_matrix, img_test):
    object_num_in_img = input_x_matrix.shape[0]

    histSize = 256
    histRange = (0, 256)
    accumulate = False

    b_hist_array = np.zeros(histSize).reshape(histSize, 1)
    g_hist_array = np.zeros(histSize).reshape(histSize, 1)
    r_hist_array = np.zeros(histSize).reshape(histSize, 1)

    for k in range(object_num_in_img):
        xmin, xmax, ymin, ymax = find_coor(input_x_matrix, img_test, k)
        roi_img = img_test[ymin: ymax, xmin: xmax]

    # xxx = plt.imshow(roi_img)
    # plt.show()
    #
    # return
#
        bgr_planes = cv.split(roi_img)
        # print(bgr_planes)
        b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

        new_b_hist = b_hist / np.amax(b_hist)
        new_g_hist = g_hist / np.amax(g_hist)
        new_r_hist = r_hist / np.amax(r_hist)

        b_hist_array = np.concatenate((b_hist_array, new_b_hist), axis=1)
        g_hist_array = np.concatenate((g_hist_array, new_g_hist), axis=1)
        r_hist_array = np.concatenate((r_hist_array, new_r_hist), axis=1)

    b_hist_array = np.delete(b_hist_array, (0), axis=1).reshape(256, object_num_in_img)
    g_hist_array = np.delete(g_hist_array, (0), axis=1).reshape(256, object_num_in_img)
    r_hist_array = np.delete(r_hist_array, (0), axis=1).reshape(256, object_num_in_img)

    return b_hist_array, g_hist_array, r_hist_array

# This function used to find out the difference between two histogram in chi-square method
# Three Inputs: 1. Histogram 1 as the main part
#               2. Histogram 2 used to compare with Histogram 1

# One Outputs:  1. The total amount of the difference as a single number

def chi_square(hist1, hist2):
    hist1_minus_hist2_elementwise = hist1 - hist2
    get_square = np.square(hist1_minus_hist2_elementwise)
    divide_hist1 = get_square / hist1
    where_are_NaNs = np.isnan(divide_hist1)
    divide_hist1[where_are_NaNs] = 0
    where_are_infs = np.isinf(divide_hist1)
    divide_hist1[where_are_infs] = 0
    sum_array = np.sum(divide_hist1)

    return sum_array

def find_diff_list_between_two_least_num(list1):
    length = len(list1)
    list1.sort()
    smallest = list1[0]
    second_smallest = list1[1]
    diff = second_smallest - smallest

    return diff
    print("Largest element is:", list1[length-1])
    print("Smallest element is:", list1[0])
    print("Second Largest element is:", list1[length-2])
    print("Second Smallest element is:", list1[1])

def find_best_similarity_object(exp_object_bhist, exp_object_ghist, exp_object_rhist, additional_img_xdata, additional_img):
    add_b_hist, add_g_hist, add_r_hist = find_bgr_hist(additional_img_xdata, additional_img)
    exp_number_object = add_b_hist.shape[1]
    sum_diff_list = []
    for i in range(exp_number_object):
        b_compareHist = chi_square(exp_object_bhist.reshape(256, 1), add_b_hist[:, i].reshape(256, 1))
        g_compareHist = chi_square(exp_object_ghist.reshape(256, 1), add_g_hist[:, i].reshape(256, 1))
        r_compareHist = chi_square(exp_object_rhist.reshape(256, 1), add_r_hist[:, i].reshape(256, 1))
        cur_sum_diff = b_compareHist + g_compareHist + r_compareHist
        sum_diff_list.append(cur_sum_diff)
    min_sum_object = sum_diff_list.index(min(sum_diff_list))
    diff_x = find_diff_list_between_two_least_num(sum_diff_list)
    xmin, xmax, ymin, ymax = find_coor(additional_img_xdata, additional_img, min_sum_object)
    roi_img = additional_img[ymin: ymax, xmin: xmax]
    # print(diff_x)
    # cv.imshow("Image", roi_img)
    # cv.waitKey(1000)
    # # cv.destroyAllWindows()
    # print(sum_diff_list)
    return min_sum_object, sum_diff_list, diff_x

# This function used to find out two similarity obejcts in two different images based on RGB images
# Four Inputs: 1. input matrix generated from YOLO output txtfile of image 1
#              2. corresponding input image in rgb format of image 1
#              3. input matrix generated from YOLO output txtfile of image 2
#              4. corresponding input image in rgb format of image 2

# Three Outputs: 1. Pair list of each object in image 1 and the most similarity object in image 2
#                2. Dictionary contains the score of each pair
#                3. List contains the difference value between two lowest score

def find_rgb_similarity(img_accord, x_accord_matrix, img_nissan, x_nissan_matrix):
    b_a, g_a, r_a = find_bgr_hist(x_accord_matrix, img_accord)
    hist_match_list = []
    least_two_diff_list = []
    hist_pair_score_dict ={}
    for i in range(x_accord_matrix.shape[0]):
        x, y, z = find_best_similarity_object(b_a[:, i], g_a[:, i],
                                        r_a[:, i], x_nissan_matrix, img_nissan)
        match_pair = [i, x]
        least_two_diff_list.append(z)
        hist_match_list.append(match_pair)
        if (i, x) not in hist_pair_score_dict:
            hist_pair_score_dict[(i, x)] = np.min(y)

    return hist_match_list, hist_pair_score_dict, least_two_diff_list

# This function used to find out two similarity obejcts in two different images based on keypoints method
# Four Inputs: 1. input matrix generated from YOLO output txtfile of image 1
#              2. corresponding input image in rgb format of image 1
#              3. input matrix generated from YOLO output txtfile of image 2
#              4. corresponding input image in rgb format of image 2

# Three Outputs: 1. Pair list of each object in image 1 and the most similarity object in image 2
#                2. Dictionary contains the number of keypoints of each pair


def find_sift_keypoint(img_accord, x_accord_matrix, img_nissan, x_nissan_matrix):
    object_num_in_img_accord = x_accord_matrix.shape[0]
    object_num_in_img_nissan = x_nissan_matrix.shape[0]
    match_list = []
    pair_score_dict = {}
# Initiate SIFT detector
    for i in range(object_num_in_img_accord):
        xmin1, xmax1, ymin1, ymax1 = find_coor(x_accord_matrix, img_accord, i)
        good_number_list = []
        for j in range(object_num_in_img_nissan):
            xmin2, xmax2, ymin2, ymax2 = find_coor(x_nissan_matrix, img_nissan, j)
            sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
            roi_img1 = img_accord[ymin1: ymax1, xmin1: xmax1]
            roi_img2 = img_nissan[ymin2: ymax2, xmin2: xmax2]
            kp1, des1 = sift.detectAndCompute(roi_img1, None)
            kp2, des2 = sift.detectAndCompute(roi_img2, None)
            # print(des1.shape)
            # print(des2.shape)
            # print(2)
            if des1 is not None and des2 is not None:
                if des2.shape[0] != 1 and des1.shape[0] != 1:
    # BFMatcher with default params
                    bf = cv.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
                    good = []
                    for m, n in matches:
                        if m.distance < 0.75*n.distance:
                            good.append([m])
                    # img3 = cv.drawMatchesKnn(roi_img1, kp1, roi_img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    # # show the keypoints result
                    # xxx = plt.imshow(img3)
                    # plt.show()
                    number_good = len(good)
                    good_number_list.append(number_good)
                else:
                    good_number_list.append(0)
            else:
                good_number_list.append(0)
        best_fit_object_index = np.argmax(good_number_list)
        match_pair = [i, best_fit_object_index]
        match_list.append(match_pair)
        if (i, best_fit_object_index) not in pair_score_dict:
            pair_score_dict[(i, best_fit_object_index)] = np.max(good_number_list)



    return match_list, pair_score_dict


least_two_list = []
save_least_two_diff_path = 'least_two.npy'
if not os.path.exists(save_least_two_diff_path):
    np.save(save_least_two_diff_path, least_two_list)
    print(1)
else:
    least_two_list = np.load(save_least_two_diff_path)
    print(least_two_list.shape)

if __name__ == "__main__":
    # Set Data range for input need to change based on the number of images
    case_num = 1            # need to be changed as the case is different
    infiniti_img_num = 378  # need to be changed as the case is different
    nissan_img_num = 225    # need to be changed as the case is different
    data_range = list(range(378, 379)) # need to be changed as the data range is changed

    number_diff = infiniti_img_num - nissan_img_num

    # case1 infiniti #378 nissan #225
    # case2 infiniti #783 nissan #623
    # case3 infiniti #1050 nissan #890
    # case4 infiniti #1421 nissan #1250
    # case5 infiniti #1674 nissan #1500

    for k in data_range:
        i = str(k)
        while len(i) != 6:
            i = '0' + str(i)
        print(i)

        j = str(k-number_diff)
        while len(j) != 6:
            j = '0' + str(j)
        print(j)

        Argument = {}
        Argument['root_directory'] = 'B:/SRIP_2019/image_similarity/8_7_test3_image/'
        Argument['number_features'] = 8
        Argument['infiniti_input_txt_file_name'] = Argument['root_directory'] + 'txt_files/' + 'results' + str(case_num) + '_infiniti.txt'
        Argument['nissan_input_txt_file_name'] = Argument['root_directory'] + 'txt_files/' + 'results' + str(case_num) + '_nissan.txt'
        Argument['txt_file_camera_holder1'] = 'infiniti'
        Argument['txt_file_camera_holder2'] = 'nissan'
        Argument['infiniti_input_img_file_name'] = Argument['root_directory'] + 'case' + str(case_num) + '/infiniti/' + i + 'rgb.png'
        Argument['nissan_input_img_file_name'] = Argument['root_directory'] + 'case' + str(case_num) + '/nissan/' + j + 'rgb.png'
        Argument['save_result_csvfile'] = 'results.csv'
        Argument['save_result_csvfile_keypoint'] = 'results_keypoint.csv'
    # Sample Test
    # Argument['root_directory'] = 'B:/SRIP_2019/image_similarity/sample_image_similarity/'
    # Argument['number_features'] = 8
    # Argument['infiniti_input_txt_file_name'] = Argument['root_directory'] + 'results_accord_sub_sync.txt'
    # Argument['nissan_input_txt_file_name'] = Argument['root_directory'] + 'results_nissan_sub_sync.txt'
    # Argument['txt_file_camera_holder1'] = 'accord'
    # Argument['txt_file_camera_holder2'] = 'nissan'
    # Argument['infiniti_input_img_file_name'] = Argument['root_directory'] + Argument['txt_file_camera_holder1'] + '_00234.png'
    # Argument['nissan_input_img_file_name'] = Argument['root_directory'] + Argument['txt_file_camera_holder2'] + '_00234.png'


        x_infiniti_hist, y_infiniti_hist = find_sample_data(Argument['infiniti_input_txt_file_name'], Argument['txt_file_camera_holder1'],
                                                    i, Argument['number_features'])
        x_nissan_hist, y_nissan_hist = find_sample_data(Argument['nissan_input_txt_file_name'], Argument['txt_file_camera_holder2'],
                                                    j, Argument['number_features'])

        img_test_infiniti_hist = cv.imread(Argument['infiniti_input_img_file_name'], cv.IMREAD_UNCHANGED)
        img_test_nissan_hist = cv.imread(Argument['nissan_input_img_file_name'], cv.IMREAD_UNCHANGED)
        x1, y1, z1 = find_rgb_similarity(img_test_infiniti_hist, x_infiniti_hist, img_test_nissan_hist, x_nissan_hist)
        print(x1)
        print(y1)
        # print(z1)

    # HSV mode Test
        hsv_x, hsv_y, hsv_z = find_hsv_similarity(x_infiniti_hist, img_test_infiniti_hist, x_nissan_hist, img_test_nissan_hist)
        hsv_x2, hsv_y2, hsv_z2 = find_h_similarity(x_infiniti_hist, img_test_infiniti_hist, x_nissan_hist, img_test_nissan_hist)
        print(hsv_x)
        print(hsv_y)
        print(hsv_z)

        print(hsv_x2)
        print(hsv_y2)
        print(hsv_z2)

        # x = np.load(save_least_two_diff_path)
        # new_x = np.append(x, z1)
        # np.save(save_least_two_diff_path, new_x)
    # Keypoint method shown as following
        x_infiniti_keypoint, y_infiniti_keypoint = find_sample_data(Argument['infiniti_input_txt_file_name'], 'infiniti', i, Argument['number_features'])
        x_nissan_keypoint, y_nissan_keypoint = find_sample_data(Argument['nissan_input_txt_file_name'], 'nissan', j, Argument['number_features'])

        img_test_infiniti_keypoint = cv.imread(Argument['infiniti_input_img_file_name'], cv.IMREAD_GRAYSCALE)
        img_test_nissan_keypoint = cv.imread(Argument['nissan_input_img_file_name'], cv.IMREAD_GRAYSCALE)
        x2, y2 = find_sift_keypoint(img_test_infiniti_keypoint, x_infiniti_keypoint, img_test_nissan_keypoint, x_nissan_keypoint)
        print(x2)
        print(y2)

        # if not os.path.exists(Argument['save_result_csvfile']):
        #     with open(Argument['save_result_csvfile'], 'wb') as csvfile:
        #         filewriter = csv.writer(csvfile, delimiter=',',
        #                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #
        # if not os.path.exists(Argument['save_result_csvfile_keypoint']):
        #     with open(Argument['save_result_csvfile_keypoint'], 'wb') as csvfile:
        #         filewriter = csv.writer(csvfile, delimiter=',',
        #                                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
        # for i in range(len(x1)):
        #     cur_csv_data = []
        #     x1_xmin, x1_xmax, x1_ymin, x1_ymax = find_coor(x_accord_hist, img_test_accord_hist, i)
        #     cur_csv_data.append(i)
        #     x2_xmin, x2_xmax, x2_ymin, x2_ymax = find_coor(x_nissan_hist, img_test_nissan_hist, x1[i][1])
        #     cur_csv_data.append(x1[i][1])
        #     cur_csv_data.append(y1[(i, x1[i][1])])
        #
        #     roi_img_x1 = img_test_accord_hist[x1_ymin: x1_ymax, x1_xmin: x1_xmax]
        #     roi_img_x2 = img_test_nissan_hist[x2_ymin: x2_ymax, x2_xmin: x2_xmax]
        #
        #     bgr_image_x1 = roi_img_x1[..., ::-1]
        #     bgr_image_x2 = roi_img_x2[..., ::-1]
        #
        #     f = plt.figure()
        #     f.add_subplot(1, 2, 1)
        #     plt.imshow(bgr_image_x1)
        #     f.add_subplot(1, 2, 2)
        #     plt.imshow(bgr_image_x2)
        #     plt.show(block=True)
        #
        #     true_or_false = input("Please give me an input (True is 1 and False is 0):")
        #
        #     cur_csv_data.append(int(true_or_false))
        #
        #     with open(Argument['save_result_csvfile'], 'a') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow(cur_csv_data)
        #     csvfile.close()
        #
        # for j in range(len(x2)):
        #     cur_csv_data = []
        #     x1_xmin, x1_xmax, x1_ymin, x1_ymax = find_coor(x_accord_keypoint, img_test_accord_keypoint, j)
        #     cur_csv_data.append(j)
        #     x2_xmin, x2_xmax, x2_ymin, x2_ymax = find_coor(x_nissan_keypoint, img_test_nissan_keypoint, x2[j][1])
        #     cur_csv_data.append(x2[j][1])
        #     cur_csv_data.append(y2[(j, x2[j][1])])
        #
        #     roi_img_x1 = img_test_accord_hist[x1_ymin: x1_ymax, x1_xmin: x1_xmax]
        #     roi_img_x2 = img_test_nissan_hist[x2_ymin: x2_ymax, x2_xmin: x2_xmax]
        #
        #     bgr_image_x1 = roi_img_x1[..., ::-1]
        #     bgr_image_x2 = roi_img_x2[..., ::-1]
        #
        #     f = plt.figure()
        #     f.add_subplot(1, 2, 1)
        #     plt.imshow(bgr_image_x1)
        #     f.add_subplot(1, 2, 2)
        #     plt.imshow(bgr_image_x2)
        #     plt.show(block=True)

            # true_or_false = input("Please give me an input (True is 1 and False is 0):")
            #
            # cur_csv_data.append(int(true_or_false))
            #
            # with open(Argument['save_result_csvfile_keypoint'], 'a') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(cur_csv_data)
            # csvfile.close()


