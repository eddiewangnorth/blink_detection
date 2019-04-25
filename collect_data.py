import cv2
from omnivision_camera import OVCamera
import time

if __name__ == '__main__':
    blink = False
    cam = OVCamera()
    success = cam.start_camera()
    output = cam.write_reg_value(AEC=420, AGC=3)
    blink_path = 'blinks'
    open_path = 'open'
    now = str(time.time())
    i = 0
    while True:
        start = time.time()
        img = cam.capture_frame()
        print(time.time()-start)
        cv2.imshow('img', img)
        cv2.waitKey(5)
        if blink:
            cv2.imwrite(blink_path + '/img_{}_{}.png'.format(now, i), img)
        else:
            cv2.imwrite(open_path + '/img_{}_{}.png'.format(now, i), img)
        i += 1

