import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,Dropout,MaxPooling2D
from keras.layers import Activation,Flatten,Dense,Dropout
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

girisverisi=np.load("girisverimiz.npy")
girisverisi=np.reshape(girisverisi,(-1,224,224,3))
cikisverisi=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]])

splitverisi=girisverisi[1:6]
splitverisi=np.append(splitverisi,girisverisi[24:29])
splitverisi=np.reshape(splitverisi,(-1,224,224,3))
splitcikis=np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[0,1]])

model=Sequential() #düzenimiz ardışık olduğundan

model.add(Conv2D(50,3,strides=(4,4),input_shape=(224,224,3)))       #default stridemız 2,2 dir
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

model.compile(loss='categorical_crossentropy',optimizer= RMSprop(lr=0.00001),metrics=['accuracy']) #binary olmasının nedeni 2 li sınıflandırma yapmamızdır.2 den fazla ise 'categorical_crossentropy' kullanılır

model.load_weights("kerasileuygulama")
def resmiklasordenal(dosyaadi):
    resim=cv2.imread("%s"%dosyaadi)
    return resim

girisverisi=np.array([])
klasordenalınanresim=0
string ='veriseti/araba.jpg' #dosyanın içindeki dosyaları alacağı için / ile biter
klasordenalınanresim=resmiklasordenal(string)
boyutlandirilmisresim=cv2.resize(klasordenalınanresim,(224,224))
girisverisi=np.append(girisverisi,boyutlandirilmisresim)

girisverisi=np.reshape(girisverisi,(-1,224,224,3))

print(model.predict(girisverisi))

