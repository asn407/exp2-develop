import cv2
import glob
import pandas as pd
import numpy as np
from sklearn import svm
import pickle

# 学習データ
N = 200

labels0 = []
labels1 = []
images0 = []
images1 = []

for i in range(N):
    # 学習しやすいよう加工
    img_0 = cv2.imread('./data_png/locking/'+str(i+1)+'.png')
    img_1 = cv2.imread('./data_png/no_locking/'+str(i+1)+'.png')
    
    img_0 = img_0.flatten()
    img_1 = img_1.flatten()

    # 画像をリストに追加
    images0.append(img_0)
    images1.append(img_1)
    labels0.append('locking')
    labels1.append('no_locking')
    print(str(i) +': is finished')

labels = labels0 + labels1
images = images0 + images1
    
# 学習モデルを作成
model = svm.SVC(decision_function_shape='ovr')
print('svm.SVC')
model.fit(images, labels)
print('fit')
print(model.score(images, labels))
print('model.score')

# pickle.dump(model, open('model.sav', 'wb'))

# テスト
for i in range(50):
    print(201+i)
    img = cv2.imread('./data_png/locking/'+str(201+i)+'.png')
    img = cv2.resize(img, (64, 64))
    # img = img.flatten()
    print(str(201+i)+" : 判定結果"+str(model.predict([img])))

print('=================')

for i in range(50):
    print(str(i+1))
    img = cv2.imread('./data_png/no_locking/'+str(201+i)+'.png')
    img = cv2.resize(img, (64, 64))
    # img = img.flatten()
    print(str(201+i)+" : 判定結果"+str(model.predict([img])))