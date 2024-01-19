# 学習データ数
train_nums = [200, 200, 120, 120]
# テストデータ数
test_nums = [50, 50, 30, 30]
# ラベル
labels = ['locking', 'silent', 'checking', 'opening']
# スコア用リスト
result = []

# ライブラリのインポート
print('importing librarys...')
import cv2
import glob
from sklearn import svm
import pickle

# モデルのインポート
print('importing the model...')
model = pickle.load(open('model.sav', 'rb'))

# 測定開始
print('please wait for a moment...')
for i in range(len(test_nums)):
    for j in range(test_nums[i]):

        # データの用意
        X_test = cv2.imread('./data_png/'+labels[i]+'/'+str(train_nums[i]+j+1)+'.png')
        X_test = cv2.resize(X_test, (480, 480))
        X_test = X_test.flatten()
        y_test = labels[i]
        result.append(0)

        # 予測
        output = model.predict([X_test])

        # 正解の場合はスコア加算
        if output == y_test:
            result[i] += 1

        print(str(j+1)+' / '+str(test_nums[i])+' is finished : '+str(output)+' / '+labels[i])

# 結果発表
total_result = 0
total_test = 0
print('==========')
print('score')
for i in range(len(labels)):
    rate = result[i] / test_nums[i] * 100
    print(labels[i]+' : '+str(rate)+' % ('+str(result[i])+'/'+str(test_nums[i])+')')
    total_result += result[i]
    total_test += test_nums[i]

rate = total_result / total_test * 100
print('total : '+str(rate)+' % ('+str(total_result)+'/'+str(total_test)+')')