import cv2
import glob
from sklearn import svm
import pickle

# モデルのオープン
model = pickle.load(open('model.sav', 'rb'))

for i in range(50):
    # print(i+201)
    img0 = cv2.imread('./data_png/locking/'+str(201+i)+'.png')
    # img = cv2.resize(img, (64, 64))
    img0 = img0.flatten()
    print(str(201+i)+" : 判定結果"+str(model.predict([img0])))

print('=================')

for i in range(50):
    # print(i+201)
    img1 = cv2.imread('./data_png/no_locking/'+str(201+i)+'.png')
    # img = cv2.resize(img, (64, 64))
    img1 = img1.flatten()
    print(str(201+i)+" : 判定結果"+str(model.predict([img1])))