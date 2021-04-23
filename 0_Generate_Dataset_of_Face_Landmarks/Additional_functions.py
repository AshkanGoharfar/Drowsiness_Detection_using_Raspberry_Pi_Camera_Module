from Operators import *




def record_camera(flag_drowsiness):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # predictor = dlib.shape_predictor(args["shapePredictor"])

    cap = cv2.VideoCapture(0)

    while True:
        try:
            _, frame = cap.read()

            image = frame
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
            distances = [flag_drowsiness, eye_left_distance_1, eye_left_distance_2, eye_right_distance_1,
                         eye_right_distance_2,
                         lip_first_layer_distance_1, lip_first_layer_distance_2, lip_first_layer_distance_3,
                         lip_first_layer_distance_4, lip_first_layer_distance_5, lip_second_layer_distance_1,
                         lip_second_layer_distance_2, lip_second_layer_distance_3]
            store_to_cvs(distances)
            cv2.imshow("Output", image)
            k = cv2.waitKey(300) & 0xFF
            cv2.destroyAllWindows()
            if k == 27:
                break
        except:
            break

    cap.release()
    cv2.destroyAllWindows()


# record_camera(0)



def read_DROZY_frames(flag_drowsiness, input_file):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    img_counter = 0
    while True:
        try:
            image = cv2.imread(input_file + str(img_counter) + '.png')
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

            distances = [flag_drowsiness, eye_left_distance_1, eye_left_distance_2, eye_right_distance_1,
                         eye_right_distance_2,
                         lip_first_layer_distance_1, lip_first_layer_distance_2, lip_first_layer_distance_3,
                         lip_first_layer_distance_4, lip_first_layer_distance_5, lip_second_layer_distance_1,
                         lip_second_layer_distance_2, lip_second_layer_distance_3]
            store_to_cvs(distances)

            cv2.imshow("Output", image)
            k = cv2.waitKey(300) & 0xFF
            cv2.destroyAllWindows()
            img_counter += 1
            if k == 27:
                break
        except:
            break
    cv2.destroyAllWindows()
