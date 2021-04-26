from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import datetime


def eye_distance(shape, image):
    shape_eyes = shape[36:48]
    # # 1 - 4, 2 - 5
    # # 7 - 10, 8 - 11
    for (x, y) in shape_eyes:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # print(shape_eyes[1], shape_eyes[4], shape_eyes[2], shape_eyes[5])
    eye_left_distance_1 = np.sqrt(
        np.square(shape_eyes[4][0] - shape_eyes[1][0]) + np.square(shape_eyes[4][1] - shape_eyes[1][1]))
    eye_left_distance_2 = np.sqrt(
        np.square(shape_eyes[5][0] - shape_eyes[2][0]) + np.square(shape_eyes[5][1] - shape_eyes[2][1]))
    eye_right_distance_1 = np.sqrt(
        np.square(shape_eyes[10][0] - shape_eyes[7][0]) + np.square(shape_eyes[10][1] - shape_eyes[7][1]))
    eye_right_distance_2 = np.sqrt(
        np.square(shape_eyes[11][0] - shape_eyes[8][0]) + np.square(shape_eyes[11][1] - shape_eyes[8][1]))
    return image, eye_left_distance_1, eye_left_distance_2, eye_right_distance_1, eye_right_distance_2


def lips_distance(shape, image):
    shape_lips = shape[48:]
    for (x, y) in shape_lips:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # 5 - 7, 4 - 8, 3 - 9, 2 - 10, 1 - 11, 15- 17, 14 - 18, 13 - 19
    lip_first_layer_distance_1 = np.sqrt(
        np.square(shape_lips[5][0] - shape_lips[7][0]) + np.square(shape_lips[5][1] - shape_lips[7][1]))
    lip_first_layer_distance_2 = np.sqrt(
        np.square(shape_lips[4][0] - shape_lips[8][0]) + np.square(shape_lips[4][1] - shape_lips[8][1]))
    lip_first_layer_distance_3 = np.sqrt(
        np.square(shape_lips[3][0] - shape_lips[9][0]) + np.square(shape_lips[3][1] - shape_lips[9][1]))
    lip_first_layer_distance_4 = np.sqrt(
        np.square(shape_lips[2][0] - shape_lips[10][0]) + np.square(shape_lips[2][1] - shape_lips[10][1]))
    lip_first_layer_distance_5 = np.sqrt(
        np.square(shape_lips[1][0] - shape_lips[11][0]) + np.square(shape_lips[1][1] - shape_lips[11][1]))
    lip_second_layer_distance_1 = np.sqrt(
        np.square(shape_lips[15][0] - shape_lips[17][0]) + np.square(shape_lips[15][1] - shape_lips[17][1]))
    lip_second_layer_distance_2 = np.sqrt(
        np.square(shape_lips[14][0] - shape_lips[18][0]) + np.square(shape_lips[14][1] - shape_lips[18][1]))
    lip_second_layer_distance_3 = np.sqrt(
        np.square(shape_lips[13][0] - shape_lips[19][0]) + np.square(shape_lips[13][1] - shape_lips[19][1]))
    return image, lip_first_layer_distance_1, lip_first_layer_distance_2, lip_first_layer_distance_3, lip_first_layer_distance_4, lip_first_layer_distance_5, lip_second_layer_distance_1, lip_second_layer_distance_2, lip_second_layer_distance_3


def store_to_cvs(distances):
    f = open('Data/Drowsiness_dataset.csv', 'a')
    f.write('\n' + str(distances[0]) + ',' + str(distances[1]) + ',' + str(distances[2]) + ',' + str(
        distances[3]) + ',' + str(distances[4]) + ',' + str(distances[5]) + ',' + str(distances[6]) + ',' + str(
        distances[7]) + ',' + str(distances[8]) + ',' + str(distances[9]) + ',' + str(distances[10]) + ',' + str(
        distances[11]) + ',' + str(distances[12]) + ',' + str(distances[13]))
    f.close()


def get_frames(video_name, min_difference):
    video_name_timestamps = 'Data/DROZY/timestamps/' + video_name
    file1 = open(video_name_timestamps, 'r')
    Lines = file1.readlines()
    count = 0
    frames_timestamps = {}
    for line in Lines:
        line = line.strip().split(' ')
        if count == 0:
            for item in line:
                if int(item[0]) == 0:
                    item.lstrip('0')
            current_time = datetime.datetime(int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]),
                                             int(line[5]))
            frames_timestamps[count] = {'time': current_time}

        else:
            for item in line:
                if int(item[0]) == 0:
                    item.lstrip('0')
            current_time = datetime.datetime(int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4]),
                                             int(line[5]))
            if str(current_time - frames_timestamps[list(frames_timestamps.keys())[-1]]['time']) == min_difference:
                frames_timestamps[count] = {'time': current_time}
        count += 1
    return frames_timestamps


def read_DROZY_videos(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    image = imutils.resize(image, width=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(image, "Face".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        image, eye_left_distance_1, eye_left_distance_2, eye_right_distance_1, eye_right_distance_2 = eye_distance(
            shape, image)
        image, lip_first_layer_distance_1, lip_first_layer_distance_2, lip_first_layer_distance_3, lip_first_layer_distance_4, lip_first_layer_distance_5, lip_second_layer_distance_1, lip_second_layer_distance_2, lip_second_layer_distance_3 = lips_distance(
            shape, image)
    distances = [eye_left_distance_1,
                 eye_left_distance_2, eye_right_distance_1,
                 eye_right_distance_2,
                 lip_first_layer_distance_1, lip_first_layer_distance_2, lip_first_layer_distance_3,
                 lip_first_layer_distance_4, lip_first_layer_distance_5, lip_second_layer_distance_1,
                 lip_second_layer_distance_2, lip_second_layer_distance_3]
    return distances
