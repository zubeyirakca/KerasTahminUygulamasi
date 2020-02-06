import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,Dropout,MaxPooling2D
from keras.layers import Activation,Flatten,Dense,Dropout
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

girisverisi=np.load("girisverimiz.npy")
girisverisi=np.reshape(girisverisi,(-1,224,224,3))
girisverisi=girisverisi/255

cikisverisi=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])

splitverisi=girisverisi[1:6]
splitverisi=np.append(splitverisi,girisverisi[24:29])
splitverisi=np.reshape(splitverisi,(-1,224,224,3))
splitcikis=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1]])

girisverisi=girisverisi/255

model=Sequential() #düzenimiz ardışık olduğundan

model.add(Conv2D(50,3,strides=(4,4),input_shape=(224,224,3)))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,3))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(MaxPooling2D((5,5)))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(Conv2D(50,2))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(50,2))

model.add(Flatten())

model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1000,activation='relu'))
model.add(Dense(2)) #2 veriden birine gidecek
model.add(Activation('softmax'))

#data augmentation kısmı: sonradan eklendi silinebilir çalışmıyor.
"""datagen=ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=True,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.2,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.2,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images)

datagen.fit(girisverisi)"""


model.compile(loss='categorical_crossentropy',optimizer= RMSprop(lr=0.00001),metrics=['accuracy']) #binary olmasının nedeni 2 li sınıflandırma yapmamızdır.2 den fazla ise 'categorical_crossentropy' kullanılır
model.summary()
print(splitverisi.shape)
model.fit_generator(datagen.flow(girisverisi, cikisverisi,
                                     batch_size=4),
                        epochs=20,
                        validation_data=(splitverisi, splitcikis),
                        workers=4)

model.save("kerasileuygulamaimgaug")