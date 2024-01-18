import librosa
import matplotlib.pyplot as plt

# データ数
N = 250

# データの格納
y = []
no_y = []
sr = []
no_sr = []
for i in range(N):
  # データの読み込み
  locking_wav_data = librosa.load('./data_wav/locking/'+str(i+1)+'.wav')
  no_locking_wav_data = librosa.load('./data_wav/no_locking/'+str(i+1)+'.wav')
  # データの格納
  y.append(locking_wav_data[0])
  sr.append(locking_wav_data[1])
  no_y.append(no_locking_wav_data[0])
  no_sr.append(no_locking_wav_data[1])

import numpy as np
import librosa.display
S = []
no_S = []
img = []
no_img = []

for i in range(N):
  S.append(librosa.feature.melspectrogram(y=y[i], sr=sr[i], n_fft=1024, win_length=256, hop_length=256))
  no_S.append(librosa.feature.melspectrogram(y=no_y[i], sr=no_sr[i], n_fft=1024, win_length=256, hop_length=256))

for i in range(N):
  no_S_dB = librosa.power_to_db(no_S[i], ref=np.max)
  no_img.append(librosa.display.specshow(no_S_dB))
  plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
  plt.savefig('./data_png/no_locking/'+str(i+1)+'.png')
  plt.close()
