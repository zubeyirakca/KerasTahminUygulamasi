import cv2
import numpy as np

def resmiklasordenal(dosyaadi):
    resim=cv2.imread("%s"%dosyaadi)
    return resim


girisverisi=np.array([])
for i in range(30):
    klasordenalınanresim=0
    i=i+1
    string ='veriseti/%s.jpg'%i #dosyanın içindeki dosyaları alacağı için / ile biter
    klasordenalınanresim=resmiklasordenal(string)
    boyutlandirilmisresim=cv2.resize(klasordenalınanresim,(224,224))
    girisverisi=np.append(girisverisi,boyutlandirilmisresim)
    print(i+1)

girisverisi=np.reshape(girisverisi,(-1,224,224,3))
np.save("girisverimiz",girisverisi)  #girisverimiz dosyasını oluşturduk



print(girisverisi)
print(girisverisi.shape)