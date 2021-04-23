import sys

from Operators import *
import time


# sys_argv = str(sys.argv)
# min_difference = sys_argv[1]
# min_difference = '0:00:01'

def run(min_difference):
    f = open('Data/Drowsiness_dataset.csv', 'w+')
    f.write(
        'time,drowsiness,eye_left_distance_1,eye_left_distance_2,eye_right_distance_1,eye_right_distance_2,lip_first_layer_distance_1,lip_first_layer_distance_2,lip_first_layer_distance_3,lip_first_layer_distance_4,lip_first_layer_distance_5,lip_second_layer_distance_1,lip_second_layer_distance_2,lip_second_layer_distance_3')
    f.close()

    file_KSS = open('Data\DROZY\KSS.txt', 'r')
    KSS_lines = file_KSS.readlines()

    for i in range(len(KSS_lines)):
        KSS_lines[i] = KSS_lines[i].split('\n')[0].split(' ')

    total_elapsed_time = 0
    for i in range(1, 15):
        for j in range(1, 4):
            if (i == 7 and j == 1) or (i == 9 and j == 1) or (i == 10 and j == 2) or (i == 12 and j == 2) or (
                    i == 12 and j == 3) or (i == 13 and j == 3):
                pass
            else:
                time1 = time.time()
                read_DROZY_videos(KSS_lines[i - 1][j - 1], str(i) + '-' + str(j) + '.mp4', min_difference)
                time2 = time.time()
                elapsed_time = time2 - time1
                total_elapsed_time += elapsed_time
                print('i and j : ', i, j)
                print('Elapsed time : ', elapsed_time)

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Total elapsed time is : ', total_elapsed_time)
