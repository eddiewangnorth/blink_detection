from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, SeparableConv2D
from keras.models import Sequential

def build_blink_CNN(input_shape):
    model = Sequential()
    model.add(SeparableConv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(SeparableConv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

