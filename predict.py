from keras.models import load_model
import cv2
from scipy.misc import imresize
import numpy as np
import time

if __name__ == '__main__':
    test_data = np.load('test_imgs.npy')
    test_labels = np.load('test_labels.npy')
    model = load_model('models/blink_0.h5')
    predictions = []
    num_correct = 0
    for i, img in enumerate(test_data):
        cv2.imshow('c', img)
        cv2.waitKey(10)
        img = img.reshape((1,200,200,1))
        start = time.time()
        pred = model.predict(img)
        print(time.time() - start)
        if pred > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
        if predictions[-1] == test_labels[i]:
            num_correct += 1

    print(num_correct / len(predictions))



