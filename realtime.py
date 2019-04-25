from omnivision_camera import OVCamera
import cv2
from keras.models import load_model
from scipy.misc import imresize
import time
import numpy as np

eye_blink_count = 0
eye_stats = [True]

if __name__ == '__main__':
    cam = OVCamera()
    success = cam.start_camera()
    output = cam.write_reg_value(AEC=420, AGC=3)
    model = load_model('models/blink_0.h5')

    while True:
        img = cam.capture_frame()

        img2 = imresize(img, 0.5) / 255
        img2 = img2.reshape((1, 200, 200, 1))
        pred = model.predict(img2)
        if pred > 0.7:
            if eye_stats[-1] == True:
                eye_stats.append(False)
        elif pred < 0.3:
            if eye_stats[-1] == False:
                eye_stats.append(True)

        font = cv2.FONT_HERSHEY_SIMPLEX
        img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(img3, str(pred[0][0]), (50, 230), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img3, 'BLINK ' + str(eye_blink_count), (50, 170), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', img3)
        cv2.waitKey(10)

        if (len(eye_stats) == 3):
            eye_blink_count += 1
            print(eye_blink_count)
            eye_stats = [True]
