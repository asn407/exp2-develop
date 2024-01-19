# 音の種類を選択
labels = ['locking', 'silent', 'checking', 'opening']
print(labels[0]+' : 0 / '+labels[1]+' : 1 / '+labels[2]+' : 2 / '+labels[3]+' : 3')
command = input('what?: ')
what = labels[int(command)]

# 音の数を選択
num = int(input('how many?: '))

# ライブラリのインポート
print('importing librarys...')
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

print('please wait for a moment...')
for i in range(num):
  # データの読み込み
  wav_data = librosa.load('./data_wav/'+what+'/'+str(i+1)+'.wav')

  # データの情報を格納
  y = wav_data[0]
  sr = wav_data[1]

  # メルスペクトログラム画像作成
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, win_length=256, hop_length=256)
  S_dB = librosa.power_to_db(S, ref=np.max)
  img = librosa.display.specshow(S_dB)

  # matplotlibでグラフ化
  # ディレクトリに保存
  plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
  plt.savefig('./data_png/'+what+'/'+str(i+1)+'.png')
  plt.close()

  print(str(i+1)+' / '+str(num)+' is finished')