from model import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import cv2
import numpy as np
from keras.models import load_model
import glob
from keras.optimizers import Adam
from scipy.misc import imresize

if __name__ == '__main__':
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    blink_imgs = glob.glob('blinks/*.png')
    open_imgs = glob.glob('open/*.png')

    for img_path in blink_imgs:
        r = np.random.rand()
        img = cv2.imread(img_path, 0)
        img = imresize(img, 0.5) / 255
        cv2.imshow('c', img)
        cv2.waitKey(10)
        if r < 0.15:
            test_data.append(img)
            test_label.append(1)
        else:
            train_data.append(img)
            train_label.append(1)

    for img_path in open_imgs:
        r = np.random.rand()
        img = cv2.imread(img_path, 0)
        img = imresize(img, 0.5) / 255
        cv2.imshow('c', img)
        cv2.waitKey(10)
        if r < 0.15:
            test_data.append(img)
            test_label.append(0)
        else:
            train_data.append(img)
            train_label.append(0)

    model = build_blink_CNN((200, 200, 1))

    model_name = 'models/blink_0.h5'
    callbacks = [
        #EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=10, verbose=1),
        ModelCheckpoint(model_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
        TensorBoard(log_dir='logs/{}'.format(model_name[7:-3], write_graph=False))
    ]

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    np.save('test_imgs.npy', test_data)
    np.save('test_labels.npy', test_label)

    # (d1, d2, d3) = train_data.shape
    # train_data = np.reshape(train_data, (d1, d2, d3, 1))
    #
    # model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    # model.fit(train_data, train_label, batch_size=32, verbose=1, epochs=40, shuffle=True,
    #        validation_split=0.2,
    #        callbacks=callbacks)

