# 学習データ数
train_nums = [200, 200, 120, 120]
# ラベル
labels = ['locking', 'silent', 'checking', 'opening']

X_train = []
y_train = []

# ライブラリのインポート
print('importing librarys...')
import cv2
import glob
import pandas as pd
import numpy as np
from sklearn import svm
import pickle

# 読み込み開始
print('please wait for a moment...')
for i in range(len(train_nums)):

    # locking -> silent -> checking -> opening
    # の順番で読み込みを行う

    print('start '+labels[i])

    for j in range(train_nums[i]):

        # 画像の読み込み
        img = cv2.imread('./data_png/'+labels[i]+'/'+str(j+1)+'.png')

        # 学習しやすいよう画像を加工
        img = cv2.resize(img, (480, 480))
        img = img.flatten()

        # 学習用変数に格納
        X_train.append(img)
        y_train.append(labels[i])

        print(str(j+1)+' / '+str(train_nums[i])+' is finished : '+labels[i])
    
# 学習モデルを作成
print('making model...')
model = svm.SVC(decision_function_shape='ovr')
model.fit(X_train, y_train)

# コサイン類似度
print('model.score')
print(model.score(X_train, y_train))

# 作成されたモデルを保存
pickle.dump(model, open('model.sav', 'wb'))